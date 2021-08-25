from typing import Union

from torch import nn
import torch
from torch.nn import functional as F
from .modules.rcab import RCAB


class Convs(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rcab: bool = False):
        super().__init__()
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        rcab and modules.append(RCAB(num_channels=out_channels, reduction=4))
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, rcab: bool = False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            Convs(in_channels, out_channels, rcab)
        )

    def forward(self, x):
        return self.layers(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, rcab: bool = False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = Convs(in_channels, out_channels, rcab)

    def forward(self, x, y):
        x = self.upsample(x)

        diffY = y.size()[2] - x.size()[2]
        diffX = y.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        x = torch.cat((x, y), dim=1)
        return self.conv(x)


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
        x = x.contiguous().view(batch_size * x.size(1) * x.size(2), -1)
        x = self.layers(x)
        x = x.view(batch_size, self.out_features, shape[1], shape[2])
        return x


class SubRCAUNet(nn.Module):
    def __init__(self, seq_len: int, patch_size: int, temporal_features: int, attention: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        factor = 2
        self.in_conv = Convs(temporal_features, 64, rcab=attention)
        self.down1 = Downsample(64, 128, rcab=attention)
        self.down2 = Downsample(128, 256, rcab=attention)
        self.down3 = Downsample(256, 512 // factor, rcab=attention)
        self.up1 = Upsample(512, 256 // factor, rcab=attention)
        self.up2 = Upsample(256, 128 // factor, rcab=attention)
        self.up3 = Upsample(128, 64 // factor, rcab=attention)
        self.out_conv = Convs(32, 1, rcab=False)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.out_conv(x)
        return x


class RCAUNet(nn.Module):
    def __init__(self, seq_len: int, patch_size: int, temporal_features: int, attention: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.feature_extractor = FeatureExtraction(seq_len, temporal_features)
        # Fang used separate UNet architectures for t1 and t2 classification.
        self.t1_net = SubRCAUNet(seq_len, patch_size, temporal_features, attention)
        self.t2_net = SubRCAUNet(seq_len, patch_size, temporal_features, attention)

    def forward(self, x):
        x = self.feature_extractor(x.transpose(1, 3))
        t1 = self.t1_net(x)
        t2 = self.t2_net(x)
        return torch.cat((t1, t2), dim=1).transpose(3, 1)
