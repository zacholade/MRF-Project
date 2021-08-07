from typing import Union

from torch import nn
import torch


class PatchSizeTest(nn.Module):
    def __init__(self, seq_len: int, patch_size: int):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size

        self.conv1ds = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32,
                      kernel_size=(21, 1, 1), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=32,
                      kernel_size=(21, 1, 1), stride=1, padding='same'),
            nn.ReLU()
        )

        self.conv1_x = self.conv_block(32, 64, kernel_size=(1, 1, 1), stride=(2, 1, 1),
                                       batch_norm=True, activation='relu', padding='valid')
        self.conv1_y = self.conv_block(64, 64, kernel_size=(15, 1, 1), stride=1,
                                       batch_norm=True, activation='relu', padding=(7, 0, 0))

        self.conv2_x = self.conv_block(64, 128, kernel_size=(1, 3, 3), stride=(2, 1, 1),
                                       batch_norm=True, activation='relu', padding='valid')
        self.conv2_y = self.conv_block(128, 128, kernel_size=(15, 1, 1), stride=1,
                                       batch_norm=True, activation='relu', padding=(7, 0, 0))

        self.out_conv = self.conv_block(128, 2, kernel_size=(75, patch_size - 2, patch_size - 2), stride=1,
                                        batch_norm=False, activation='none', padding='valid')

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = self.conv1ds(x)

        x = self.conv1_x(x)
        y = self.conv1_y(x)
        x = x + y
        x = self.conv2_x(x)
        y = self.conv2_y(x)
        x = x + y
        x = self.out_conv(x).view(batch_size, -1)
        return x

    def conv_block(self, in_channels, out_channels, kernel_size, stride: int, padding: str,
                   batch_norm: bool, activation: Union[str, None] = 'none'):
        modules = [
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding)]
        batch_norm and modules.append(nn.BatchNorm3d(out_channels))
        activation == 'relu' and modules.append(nn.ReLU())
        return nn.Sequential(*modules)