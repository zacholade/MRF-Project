from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, num_channels: int, reduction: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, num_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // reduction, num_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.layers(x)


class RCAB(nn.Module):
    def __init__(self, num_channels, reduction: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            ChannelAttention(num_channels, reduction)
        )

    def forward(self, x):
        return x + self.layers(x)
