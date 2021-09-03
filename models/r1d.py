from torch import nn as nn


class TemporalResLayer(nn.Module):
    """Code which has been adapted from: https://github.com/irhum/R2Plus1D-PyTorch"""

    def __init__(self, in_channels, out_channels, conv1_kernel_size, conv2_kernel_size,
                 temporal_compress=False):
        super().__init__()
        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.
        block = nn.Conv1d
        # no pooling layers are used inside ResNet
        self.temporal_compress = temporal_compress

        # to allow for SAME padding
        conv1_padding = conv1_kernel_size // 2
        conv2_padding = conv2_kernel_size // 2

        if temporal_compress:
            t_stride = 2 if temporal_compress else 1
            # downsample with stride =2 the input x
            self.compress_conv = block(in_channels, out_channels, 1, stride=t_stride)
            self.compress_bn = nn.BatchNorm1d(out_channels)

            # down sample with stride = 2 for temporal or 3 for spatial when producing the residual
            self.conv1 = block(in_channels, out_channels, conv1_kernel_size,
                               padding=conv1_padding, stride=t_stride)
        else:
            self.conv1 = block(in_channels, out_channels, conv1_kernel_size, padding=conv1_padding)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = block(out_channels, out_channels, conv2_kernel_size, padding=conv2_padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        if self.temporal_compress:
            x = self.compress_bn(self.compress_conv(x))

        return self.outrelu(x + y)


class R1D(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        conv_block = nn.Conv1d
        self.conv1 = nn.Sequential(conv_block(in_channels=1, out_channels=16,
                                              kernel_size=7, stride=2, padding=3),
                                   nn.ReLU(inplace=True))

        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = TemporalResLayer(16, 16, 3, 3)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = TemporalResLayer(16, 32, 3, 3, temporal_compress=True)
        self.conv4 = TemporalResLayer(32, 64, 3, 3, temporal_compress=True)
        self.conv5 = TemporalResLayer(64, 128, 3, 3, temporal_compress=True)
        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(128, 2)

    def forward(self, x):
        # Move temporal dimension that is currently a channel to its own dimension
        x = x.unsqueeze(1)

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