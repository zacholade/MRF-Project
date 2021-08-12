import torch
import torch.nn as nn

from .modules.factorised_spatiotemporal_conv import FactorisedSpatioTemporalConv
from .modules.non_local_block import NonLocalBlock1D, NonLocalBlock2D, NonLocalBlock3D


class SpatioTemporalResLayer(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            compress (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, compress=False,
                 block=FactorisedSpatioTemporalConv):
        super().__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.compress = compress

        # to allow for SAME padding
        padding = kernel_size // 2

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
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        if self.compress:
            x = self.compress_bn(self.compress_conv(x))

        return self.outrelu(x + y)


class R2Plus1D(nn.Module):
    def __init__(self, patch_size: int, seq_len: int, factorise: bool = True):
        """
        factorise: Whether to factorise spatial and temporal dimensions.
        If false, the resulting model will be a standard residual 3d CNN.
        """
        super().__init__()
        self.patch_size = patch_size

        conv_block = FactorisedSpatioTemporalConv if factorise else nn.Conv3d
        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = conv_block(1, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 3, 3))
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(16, 16, 3, block=conv_block)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(16, 32, 3, compress=True, block=conv_block)
        self.conv4 = SpatioTemporalResLayer(32, 64, 3, compress=True, block=conv_block)
        self.conv5 = SpatioTemporalResLayer(64, 128, 3, compress=True, block=conv_block)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.linear = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x).view(-1, 128)
        x = self.linear(x)
        return x

