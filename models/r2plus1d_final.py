import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
from .modules.factorised_spatiotemporal_conv import FactorisedSpatioTemporalConv
from .modules.non_local_block import NonLocalBlock1D, NonLocalBlock2D, NonLocalBlock3D


class SpatioTemporalResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 spatial_compress=False, temporal_compress=False,
                 block=FactorisedSpatioTemporalConv):
        super().__init__()

        kernel_size = _triple(kernel_size)
        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.spatial_compress = spatial_compress
        self.temporal_compress = temporal_compress

        # to allow for SAME padding
        padding = kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2

        if self.spatial_compress or self.temporal_compress:
            # downsample with stride =2 the input x
            compress_stride_d = 2 if self.temporal_compress else 1
            compress_stride_hw = 3 if self.spatial_compress else 1
            compress_stride = (compress_stride_d, compress_stride_hw, compress_stride_hw)
            self.compress_conv = block(in_channels, out_channels, 1, stride=compress_stride)
            self.compress_bn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2when producing the residual
            self.conv1 = block(in_channels, out_channels, kernel_size, padding=padding, stride=compress_stride)
        else:
            self.conv1 = block(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # Second conv we use a large kernel size and padding to allow for same. Inspired by success of song's network.
        # standard conv->batchnorm->ReLU
        self.conv2 = block(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        if self.spatial_compress or self.temporal_compress:
            x = self.compress_bn(self.compress_conv(x))

        return self.outrelu(x + y)


class R2Plus1DFinal(nn.Module):
    def __init__(self, patch_size: int, seq_len, factorise: bool = True):
        """
        factorise: Whether to factorise spatial and temporal dimensions.
        If false, the resulting model will be a standard residual 3d CNN.
        """
        super().__init__()
        self.patch_size = patch_size

        conv_block = FactorisedSpatioTemporalConv if factorise else nn.Conv3d
        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv0 = nn.Sequential(
            conv_block(1, 16, kernel_size=(21, 3, 3), stride=(1, 1, 1), padding=(10, 1, 1)),
            conv_block(16, 16, kernel_size=(11, 3, 3), stride=(1, 1, 1), padding=(5, 1, 1)),
        )
        self.nloc_0 = NonLocalBlock3D(in_channels=16, compression=1)
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv1 = SpatioTemporalResLayer(16, 32, 3, temporal_compress=True, block=conv_block)
        self.nloc_1 = NonLocalBlock3D(in_channels=32, compression=1)
        # each of the final three layers doubles num_channels, while performing downsampling inside the first block
        self.conv2 = SpatioTemporalResLayer(32, 64, 3, temporal_compress=True, block=conv_block)
        self.nloc_2 = NonLocalBlock3D(in_channels=64, compression=1)
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, temporal_compress=True, block=conv_block)
        self.nloc_3 = NonLocalBlock3D(in_channels=128, compression=1)
        self.conv4 = SpatioTemporalResLayer(128, 256, (3, 1, 1), spatial_compress=True, temporal_compress=True, block=conv_block)
        self.nloc_4 = NonLocalBlock3D(in_channels=256, compression=1)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.linear = nn.Linear(256, 2)

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.conv0(x.unsqueeze(1))
        x = self.nloc_0(x)
        x = self.conv1(x)
        x = self.nloc_1(x)
        x = self.conv2(x)
        x = self.nloc_2(x)
        x = self.conv3(x)
        x = self.nloc_3(x)
        x = self.conv4(x)
        x = self.nloc_4(x)
        x = self.pool(x).view(-1, 256)
        x = self.linear(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal(m.weight, mode='fan_in')

# class R2Plus1DFinal(nn.Module):
#     def __init__(self, patch_size: int, seq_len: int, factorise: bool = True):
#         """
#         factorise: Whether to factorise spatial and temporal dimensions.
#         If false, the resulting model will be a standard residual 3d CNN.
#         """
#         super().__init__()
#         self.patch_size = patch_size
#         conv_block = FactorisedSpatioTemporalConv if factorise else nn.Conv3d
#
#         # Initial layers process temporal dimension exclusively.
#         self.conv_ins = nn.Sequential(
#             conv_block(1, 16, kernel_size=(15, 1, 1), stride=(1, 1, 1), padding=(7, 0, 0)),
#             nn.ReLU(inplace=True),
#             conv_block(16, 16, kernel_size=(15, 1, 1), stride=(1, 1, 1), padding=(7, 0, 0)),
#             nn.ReLU(inplace=True)
#         )
#         # output of conv2 is same size as of conv1, no downsampling/compression needed. kernel_size 3x3x3
#         self.conv1 = SpatioTemporalResLayer(16, 32, (3, 3, 3), temporal_compress=True, block=conv_block)
#         self.nloc_1 = NonLocalBlock3D(in_channels=32, compression=1)
#         # each of the final three layers doubles num_channels, while performing downsampling in temporal dimension.
#         # Same with spatial except the last as it is already 1 at this point.
#         # inside the first block
#         self.conv2 = SpatioTemporalResLayer(32, 64, kernel_size=(3, 3, 3), spatial_compress=False, temporal_compress=True, block=conv_block)
#         self.nloc_2 = NonLocalBlock3D(in_channels=64, compression=1)
#
#         self.conv3 = SpatioTemporalResLayer(64, 128, kernel_size=(3, 3, 3), spatial_compress=True, temporal_compress=True, block=conv_block)
#         self.nloc_3 = NonLocalBlock3D(in_channels=128, compression=1)
#
#         # With patch size = 7, at this last layer, W and H are now both 1. Compressed completely in spatial dimension.
#         self.conv4 = SpatioTemporalResLayer(128, 256, kernel_size=(3, 1, 1), spatial_compress=False, temporal_compress=True, block=conv_block)
#         self.nloc_4 = NonLocalBlock3D(in_channels=256, compression=1)
#         # global average pooling of the output
#         self.pool = nn.AdaptiveAvgPool3d(1)
#         self.linear = nn.Linear(256, 2)
#
#         self.apply(self._init_weights)
#
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.conv_ins(x)
#         x = self.conv1(x)
#         x = self.nloc_1(x)
#         x = self.conv2(x)
#         x = self.nloc_2(x)
#         x = self.conv3(x)
#         x = self.nloc_3(x)
#         x = self.conv4(x)
#         x = self.nloc_4(x)
#         x = self.pool(x).view(-1, 256)
#         x = self.linear(x)
#         return x
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Conv3d):
#             nn.init.kaiming_normal(m.weight, mode='fan_in')
