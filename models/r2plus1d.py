import torch.nn as nn
from torch.nn.modules.utils import _triple

from .modules.cbam import CBAMChannelReduction
from .modules.factorised_spatiotemporal_conv import FactorisedSpatioTemporalConv
from .modules.non_local_block import NonLocalBlock3D, NonLocalAttention1DFor3D


class NonLocalLevel:
    NONE = 0
    TEMPORAL = 1
    SPATIOTEMPORAL = 2


class DimensionalityReduction:
    NONE = 0
    CBAM = 1
    LINEAR = 2


class FeatureExtraction(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        interm_features = max(in_features // 2, out_features)
        self.out_features = out_features
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=interm_features),
            nn.BatchNorm1d(interm_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=interm_features, out_features=out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        batch_size = x.size(0)
        shape = x.shape
        x = x.contiguous().view(batch_size * x.size(2) * x.size(3), -1)
        x = self.layers(x)
        x = x.view(batch_size, self.out_features, shape[2], shape[3])
        return x, None


class SpatioTemporalResLayer(nn.Module):
    """Code which has been adapted from: https://github.com/irhum/R2Plus1D-PyTorch"""
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


class R2Plus1D(nn.Module):
    def __init__(self, patch_size: int, seq_len: int, factorise: bool = True,
                 dimensionality_reduction_level: int = 0, non_local_level: int = 0):
        """
        factorise: Whether to factorise spatial and temporal dimensions.
            If false, the resulting model will be a standard residual 3d CNN.
        dimensionality_reduction_level: Applies dimensionality reduction
            according to the value provided.
        non_local_level: Integrates non-local attention into the model. If dimensionality reduction
            takes place, additional non-local blocks are added after conv 1 and 2.
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
            self.dimensionality_reduction = CBAMChannelReduction(seq_len, reduction=64)
        else:  # 2
            self.dimensionality_reduction = FeatureExtraction(seq_len, 64)

        # If we are using dimensionality reduction we dont need to stride first conv to
        # downsample the temporal input.
        conv1_stride = (1, 1, 1) if self.dimensionality_reduction_level > 0 else (2, 1, 1)

        self.conv1 = nn.Sequential(conv_block(in_channels=1, out_channels=16,
                                              kernel_size=(7, 3, 3), stride=conv1_stride, padding=(3, 1, 1)),
                                   nn.ReLU(inplace=True))

        if self.use_non_local and self.dimensionality_reduction_level > 0:
            self.nloc_1 = non_local(16, compression=1)
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(16, 16, (3, 3, 3), (3, 3, 3), block=conv_block)
        if self.use_non_local and self.dimensionality_reduction_level > 0:
            self.nloc_2 = non_local(16, compression=1)
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
        orig = x.detach().clone()

        # Channel reduction if using it.
        if self.use_dimensionality_reduction:
            x, scale = self.dimensionality_reduction(x)

        # Move temporal dimension that is currently a channel to its own dimension
        x = x.unsqueeze(1)

        # Apply non local between convs if using it
        # Last 3 conv layers do temporal downsampling with very final also doing spatial.
        if self.use_non_local:
            x = self.conv1(x)
            if self.dimensionality_reduction_level > 0:
                x = self.nloc_1(x)
            x = self.conv2(x)
            if self.dimensionality_reduction_level > 0:
                x = self.nloc_2(x)
            x = self.conv3(x)
            x = self.nloc_3(x)
            # plot_1d_nlocal_attention(scale, orig)
            # scale_ = scale.view(batch_size, 3, 3, 75, 3, 3, 75).detach().cpu().numpy()
            # scale_ = scale_.mean(axis=(3, 6))
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(3, 3, figsize=(12, 7))
            # b = np.random.randint(0, batch_size-1)
            # for x_ in range(3):
            #     for y_ in range(3):
            #         ax[x_][y_].matshow(scale_[0, x_, y_], vmin=0.001, vmax=0.003)
            # plt.show()

            x = self.conv4(x)
            x = self.nloc_4(x)
            # plot_1d_nlocal_attention(scale, orig)
            x = self.conv5(x)
            x = self.nloc_5(x)
            # plot_1d_nlocal_attention(scale, orig)
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


