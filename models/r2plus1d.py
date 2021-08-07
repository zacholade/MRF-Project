import torch
import torch.nn as nn

from .modules.factorised_spatiotemporal_conv import FactorisedSpatioTemporalConv
from .modules.non_local_block import NonLocalBlock1D, NonLocalBlock2D, NonLocalBlock3D


class ChannelAttention(nn.Module):
    def __init__(self, num_channels: int, reduction: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(num_channels, num_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_channels // reduction, num_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.layers(x)


class RCAB3D(nn.Module):
    def __init__(self, num_channels, reduction: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
            ChannelAttention(num_channels, reduction)
        )

    def forward(self, x):
        return x + self.layers(x)


class SpatioTemporalResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, compress=False,
                 block=FactorisedSpatioTemporalConv, rcab: bool = False):
        super().__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.compress = compress

        # to allow for SAME padding
        padding = kernel_size // 2

        self.use_rcab = rcab
        if self.use_rcab:
            self.rcab = RCAB3D(num_channels=in_channels, reduction=4)

        if self.compress:
            # downsample with stride =2 the input x
            self.compress_conv = block(in_channels, out_channels, 1, stride=2)
            self.compress_bn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2when producing the residual
            self.conv1 = block(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = block(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = block(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        if self.use_rcab:
            x = self.rcab(x)

        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        if self.compress:
            x = self.compress_bn(self.compress_conv(x))

        return self.outrelu(x + y)


class CBAM(nn.Module):
    def __init__(self, in_channels: int,  reduction_ratio: int = 1):
        super().__init__()
        self.interm_channels = in_channels // reduction_ratio
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # The shared linear layer
        self.linear = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=self.interm_channels),
            nn.ReLU(),
            nn.Linear(in_features=self.interm_channels, out_features=in_channels),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.linear(self.avg_pool(x).squeeze(2).squeeze(2))
        max_out = self.linear(self.max_pool(x).squeeze(2).squeeze(2))
        attention_weights = self.sig(avg_out + max_out).view(x.shape[0], x.shape[1], 1, 1)
        x *= attention_weights
        return x, attention_weights.view(x.shape[0], x.shape[1])


class R2Plus1D(nn.Module):
    def __init__(self, patch_size: int, seq_len: int, factorise: bool = True,
                 cbam: bool = False, rcab: bool = False):
        """
        factorise: Whether to factorise spatial and temporal dimensions.
        If false, the resulting model will be a standard residual 3d CNN.
        """
        super().__init__()
        self.patch_size = patch_size

        self.use_cbam = cbam
        if self.use_cbam:
            self.cbam = CBAM(seq_len, reduction_ratio=1)

        conv_block = FactorisedSpatioTemporalConv if factorise else nn.Conv3d
        # first conv, with stride 1x2x2. Kernel size 3x5x5 modified from 3x7x7.
        self.conv1 = conv_block(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 3, 3))
        # output of conv2 is same size as of conv1, no downsampling/compression needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(32, 32, 3, block=conv_block, rcab=rcab)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(32, 64, 3, compress=True, block=conv_block, rcab=rcab)
        self.conv4 = SpatioTemporalResLayer(64, 128, 3, compress=True, block=conv_block, rcab=rcab)
        # With patch size = 7, at this last layer, W and H are now both 1. Compressed completely in spatial dimension.
        self.conv5 = SpatioTemporalResLayer(128, 256, 3, compress=True, block=conv_block, rcab=rcab)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.linear = nn.Linear(256, 2)

    def forward(self, x):
        if self.use_cbam:
            x, attention_weights = self.cbam(x)
        x = self.conv1(x.unsqueeze(1))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x).view(-1, 256)
        x = self.linear(x)
        return x, attention_weights if self.use_cbam else x
