import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple

from .modules.factorised_spatiotemporal_conv import FactorisedSpatioTemporalConv
from .modules.non_local_block import NonLocalBlock1D, NonLocalBlock2D, NonLocalBlock3D, NonLocalAttention1DFor3D


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
            # t_stride = 2 if temporal_compress else 1
            # s_stride = 3 if spatial_compress else 1
            # # downsample with stride =2 the input x
            # self.compress_conv = block(in_channels, out_channels, 1, stride=(t_stride, s_stride, s_stride))
            # self.compress_bn = nn.BatchNorm3d(out_channels)
            #
            # # down sample with stride = 2 for temporal or 3 for spatial when producing the residual
            # self.conv1 = block(in_channels, out_channels, conv1_kernel_size,
            #                    padding=conv1_padding, stride=(t_stride, s_stride, s_stride))
            t_kernel_size = 2 if temporal_compress else 1
            s_kernel_size = 3 if spatial_compress else 1
            self.maxpool = nn.MaxPool3d((t_kernel_size, s_kernel_size, s_kernel_size))
        else:
            ...
        self.conv1 = block(in_channels, out_channels, conv1_kernel_size, padding=conv1_padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = block(in_channels, out_channels, conv2_kernel_size, padding=conv2_padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        y = self.relu1(self.conv1(x))
        x = self.relu1(self.conv2(x))

        x = self.outrelu(x + y)
        if self.spatial_compress or self.temporal_compress:
            # x = self.compress_bn(self.compress_conv(x))
            x = self.maxpool(x)
        return x


class NonLocalLevel:
    NONE = 0
    TEMPORAL = 1
    SPATIOTEMPORAL = 2


class R2Plus1D(nn.Module):
    def __init__(self, patch_size: int, seq_len: int, factorise: bool = True, non_local_level: int = 0):
        """
        factorise: Whether to factorise spatial and temporal dimensions.
        If false, the resulting model will be a standard residual 3d CNN.
        """
        super().__init__()
        self.patch_size = patch_size

        non_local = None if non_local_level == 0 else \
            NonLocalAttention1DFor3D if non_local_level == 1 else \
            NonLocalBlock3D
        self.use_non_local = True if non_local_level > 0 else False

        conv_block = FactorisedSpatioTemporalConv if factorise else nn.Conv3d
        # first conv, with stride 1x2x2 and kernel size 3x7x7

        self.conv1 = nn.Sequential(
            conv_block(in_channels=1, out_channels=16,
                       kernel_size=(21, 3, 3), stride=(2, 1, 1), padding=(10, 1, 1)),
            nn.ReLU()
        )

        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(16, 16, (3, 3, 3), (5, 3, 3), block=conv_block)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(16, 32, (3, 3, 3), (5, 3, 3), temporal_compress=True, block=conv_block)
        if non_local:
            self.nloc_3 = non_local(32, compression=1)
        self.conv4 = SpatioTemporalResLayer(32, 64, (3, 3, 3), (5, 3, 3), temporal_compress=True, block=conv_block)
        if non_local:
            self.nloc_4 = non_local(64, compression=1)
        self.conv5 = SpatioTemporalResLayer(64, 128, (3, 3, 3), (5, 3, 3),
                                            temporal_compress=True, spatial_compress=True, block=conv_block)
        if non_local:
            self.nloc_5 = non_local(128, compression=1)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.linear = nn.Linear(128, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_non_local:
            x = self.conv3(x)
            x = self.nloc_3(x)
            x = self.conv4(x)
            x = self.nloc_4(x)
            x = self.conv5(x)
            x = self.nloc_5(x)
        else:
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)

        x = self.pool(x).view(-1, 128)
        x = self.linear(x)
        return x
