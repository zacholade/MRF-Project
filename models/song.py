from torch import nn


class NonlocalBlock(nn.Module):
    def __init__(self, intermediate_dim=None, compression=2,
                 mode='embedded', add_residual=True):
        super().__init__()


class ResidualLayer(nn.Module):
    def __init__(self):
        super().__init__()


class Song(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.conv1ds = nn.Sequential(
            nn.Conv1d(in_channels=seq_len, out_channels=16,
                      kernel_size=21, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=seq_len, out_channels=16,
                      kernel_size=21, stride=1, padding='same'),
            nn.ReLU(),
            NonlocalBlock(compression=1, mode="embedded"),
            nn.MaxPool1d(2)

        )

        self.apply(self._init_weights)

    def forward(self, x):
        ...

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal(m.weight, mode='fan_in')

    def resnet_layer(self):
        ...