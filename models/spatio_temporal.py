from typing import Union

import torch
from torch import nn


class SpatioTemporal(nn.Module):
    def __init__(self, seq_len: int = 1000, patch_size: int = 3):
        super().__init__()
        self.conv1x1 = self.conv_block(in_channels=seq_len, out_channels=200,
                                       kernel_size=1, stride=1, padding='valid',
                                       dropout=False, batch_norm=True, activation="relu")

        self.conv3x3 = self.conv_block(in_channels=200, out_channels=32,
                                       kernel_size=3, stride=1, padding='valid',
                                       dropout=False, batch_norm=True, activation="relu")

        self.conv5x5 = self.conv_block(in_channels=200, out_channels=32,
                                       kernel_size=5, stride=1, padding='valid',
                                       dropout=False, batch_norm=True, activation="relu")

        self.conv_out = self.conv_block(in_channels=64, out_channels=2,
                                        kernel_size=1, stride=1, padding='valid',
                                        dropout=False, batch_norm=False, activation=None)

        self.patch_size = patch_size

    def conv_block(self, in_channels, out_channels, kernel_size, stride: int, padding: str,
                   dropout: bool, batch_norm: bool, activation: Union[str, None] = 'none'):
        modules = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding)]
        dropout and modules.append(nn.Dropout2d(p=0.2))
        batch_norm and modules.append(nn.BatchNorm2d(out_channels))
        activation == 'relu' and modules.append(nn.ReLU())
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv1x1(x)
        conv3x3 = self.conv3x3(x[:, :, 1:4, 1:4])
        conv5x5 = self.conv5x5(x)
        concat = torch.cat((conv3x3, conv5x5), 1)
        out = self.conv_out(concat)
        out = out.squeeze(2).squeeze(2)
        return out
