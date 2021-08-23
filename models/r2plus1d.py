import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple

from .modules.cbam import CBAM, CBAMChannelReduction
from .modules.factorised_spatiotemporal_conv import FactorisedSpatioTemporalConv
from .modules.non_local_block import NonLocalBlock1D, NonLocalBlock2D, NonLocalBlock3D, NonLocalAttention1DFor3D
from .modules.util import batched_index_select


class SpatioTemporalResLayer(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            compress (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, conv1_kernel_size, conv2_kernel_size,
                 temporal_compress=False, spatial_compress=False,
                 block=FactorisedSpatioTemporalConv):
        super().__init__()
        conv1_kernel_size = _triple(conv1_kernel_size)
        conv2_kernel_size = _triple(conv2_kernel_size)
        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.temporal_compress = temporal_compress
        self.spatial_compress = spatial_compress

        # to allow for SAME padding
        conv1_padding = conv1_kernel_size[0] // 2, conv1_kernel_size[1] // 2, conv1_kernel_size[2] // 2
        conv2_padding = conv2_kernel_size[0] // 2, conv2_kernel_size[1] // 2, conv2_kernel_size[2] // 2

        if temporal_compress or spatial_compress:
            t_stride = 2 if temporal_compress else 1
            s_stride = 3 if spatial_compress else 1
            # downsample with stride =2 the input x
            self.compress_conv = block(in_channels, out_channels, 1, stride=(t_stride, s_stride, s_stride))
            self.compress_bn = nn.BatchNorm3d(out_channels)

            # down sample with stride = 2 for temporal or 3 for spatial when producing the residual
            self.conv1 = block(in_channels, out_channels, conv1_kernel_size,
                               padding=conv1_padding, stride=(t_stride, s_stride, s_stride))
        else:
            self.conv1 = block(in_channels, out_channels, conv1_kernel_size, padding=conv1_padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = block(out_channels, out_channels, conv2_kernel_size, padding=conv2_padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        if self.spatial_compress or self.temporal_compress:
            x = self.compress_bn(self.compress_conv(x))

        return self.outrelu(x + y)


class NonLocalLevel:
    NONE = 0
    TEMPORAL = 1
    SPATIOTEMPORAL = 2


class DimensionalityReduction:
    NONE = 0
    CBAM = 1
    LINEAR = 2


class R2Plus1D(nn.Module):
    def __init__(self, patch_size: int, seq_len: int, factorise: bool = True,
                 dimensionality_reduction_level: int = 0, non_local_level: int = 0):
        """
        factorise: Whether to factorise spatial and temporal dimensions.
        If false, the resulting model will be a standard residual 3d CNN.
        """
        super().__init__()
        self.patch_size = patch_size

        conv_block = FactorisedSpatioTemporalConv if factorise else nn.Conv3d

        self.use_non_local = True if non_local_level > 0 else False
        if non_local_level == 0:
            non_local = None
        elif non_local_level == 1:
            non_local = NonLocalAttention1DFor3D
        else:  # 2
            non_local = NonLocalBlock3D

        self.use_dimensionality_reduction = True if dimensionality_reduction_level > 0 else False
        self.dimensionality_reduction_level = dimensionality_reduction_level
        if dimensionality_reduction_level == 0:
            self.dimensionality_reduction = None
        elif dimensionality_reduction_level == 1:
            self.dimensionality_reduction = CBAMChannelReduction(seq_len, reduction=128)
        else:  # 2
            raise NotImplementedError('todo')

        self.conv1 = nn.Sequential(
            conv_block(in_channels=1, out_channels=16,
                       kernel_size=(7, 3, 3), stride=(2, 1, 1), padding=(3, 1, 1)),
            nn.ReLU(inplace=True)
        )

        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(16, 16, (3, 3, 3), (3, 3, 3), block=conv_block)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(16, 32, (3, 3, 3), (3, 3, 3), temporal_compress=True, block=conv_block)
        if self.use_non_local:
            self.nloc_3 = non_local(32, compression=1)
        self.conv4 = SpatioTemporalResLayer(32, 64, (3, 3, 3), (3, 3, 3), temporal_compress=True, block=conv_block)
        if self.use_non_local:
            self.nloc_4 = non_local(64, compression=1)
        self.conv5 = SpatioTemporalResLayer(64, 128, (3, 3, 3), (3, 3, 3),
                                            temporal_compress=True, spatial_compress=True, block=conv_block)
        if self.use_non_local:
            self.nloc_5 = non_local(128, compression=1)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.linear = nn.Linear(128, 2)

    def forward(self, x):
        batch_size = x.size(0)

        # Channel reduction if using it.
        if self.use_dimensionality_reduction:
            x, scale = self.dimensionality_reduction(x)

        # Move temporal dimension that is currently a channel to its own dimension
        x = x.unsqueeze(1)

        # Apply non local between convs if using it
        if self.use_non_local:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.nloc_3(x)
            x = self.conv4(x)
            x = self.nloc_4(x)
            x = self.conv5(x)
            x = self.nloc_5(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)

        # Space time average pooling. Returns shape of (Batch x 128 (num channels))
        x = self.pool(x).view(-1, 128)

        # Linear layer for classification of the central pixel.
        x = self.linear(x)
        return x
