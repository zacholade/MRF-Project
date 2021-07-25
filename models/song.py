from torch import nn


class Song(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(out_channels=16, kernel_size=21, stride=1, padding='same')
        nn.init.kaiming_normal(self.conv1d.weight, mode='fan_in')

    def forward(self, x):
        ...

    def resnet_layer(self):
        ...