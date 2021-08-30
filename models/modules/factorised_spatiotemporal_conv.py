import math

import torch.nn as nn
from torch.nn.modules.utils import _triple


class FactorisedSpatioTemporalConv(nn.Module):
    """Code which has been adapted from: https://github.com/irhum/R2Plus1D-PyTorch"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # M_i = \frac{td^2 N_{i-1} N_i}{d^2 N_{i-1} + t N_i}
        intermed_channels = max(int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) /
                                (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels))), 1)

        spatial_kernel_size = 1, kernel_size[1], kernel_size[2]
        spatial_stride = 1, stride[1], stride[2]
        spatial_padding = 0, padding[1], padding[2]

        self.spatial_conv = nn.Conv3d(in_channels=in_channels, out_channels=intermed_channels,
                                      kernel_size=spatial_kernel_size, stride=spatial_stride,
                                      padding=spatial_padding)
        self.bn = nn.BatchNorm3d(intermed_channels)

        # 2D + 1 advantage. Additional ReLU between 2d and 1d convolution doubles the number
        # of non-linearities. Thus, increasing complexity of the functions it can represent.
        self.relu = nn.ReLU()

        temporal_kernel_size = kernel_size[0], 1, 1
        temporal_stride = stride[0], 1, 1
        temporal_padding = padding[0], 0, 0
        self.temporal_conv = nn.Conv3d(in_channels=intermed_channels, out_channels=out_channels,
                                       kernel_size=temporal_kernel_size, stride=temporal_stride,
                                       padding=temporal_padding)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x
