from torch import nn

from models.modules.non_local_block import NonLocalBlock1D


class ResNetLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 16,
                 kernel_size: int = 3, stride: int = 1,
                 activation: bool = True,
                 batch_normalization: bool = True):
        super().__init__()
        modules = [nn.Conv1d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride, padding='same')]
        batch_normalization and modules.append(nn.BatchNorm1d(out_channels))
        activation and modules.append(nn.ReLU())
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class Song(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.conv1ds = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16,
                      kernel_size=21, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16,
                      kernel_size=21, stride=1, padding='same'),
            nn.ReLU())

        self.nloc_0 = NonLocalBlock1D(in_channels=16, compression=1)

        self.maxp_1 = nn.MaxPool1d(2)
        self.resx_1 = ResNetLayer(in_channels=16, out_channels=32,
                                  kernel_size=1, activation=True,
                                  batch_normalization=False)
        self.resy_1 = ResNetLayer(in_channels=32, out_channels=32,
                                  kernel_size=21, activation=True,
                                  batch_normalization=False)
        self.nloc_1 = NonLocalBlock1D(in_channels=32, compression=1)

        self.maxp_2 = nn.MaxPool1d(2)
        self.resx_2 = ResNetLayer(in_channels=32, out_channels=64,
                                  kernel_size=1, activation=True,
                                  batch_normalization=False)
        self.resy_2 = ResNetLayer(in_channels=64, out_channels=64,
                                  kernel_size=21, activation=True,
                                  batch_normalization=False)
        self.nloc_2 = NonLocalBlock1D(in_channels=64, compression=1)

        self.maxp_3 = nn.MaxPool1d(2)
        self.resx_3 = ResNetLayer(in_channels=64, out_channels=128,
                                  kernel_size=1, activation=True,
                                  batch_normalization=False)
        self.resy_3 = ResNetLayer(in_channels=128, out_channels=128,
                                  kernel_size=21, activation=True,
                                  batch_normalization=False)
        self.nloc_3 = NonLocalBlock1D(in_channels=128, compression=1)

        self.out = nn.Linear(in_features=128, out_features=2)
        self.apply(self._init_weights)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, -1)
        x = self.conv1ds(x)
        x = self.nloc_0(x)

        x = self.maxp_1(x)
        x = self.resx_1(x)
        y = self.resy_1(x)
        x = self.nloc_1(x + y)

        x = self.maxp_2(x)
        x = self.resx_2(x)
        y = self.resy_2(x)
        x = self.nloc_2(x + y)

        x = self.maxp_3(x)
        x = self.resx_3(x)
        y = self.resy_3(x)
        x = self.nloc_3(x + y)
        x = x.mean(2)
        x = self.out(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal(m.weight, mode='fan_in')
