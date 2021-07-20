import numpy as np
import torch
from torch import nn as nn
from sub_layers import *

class CohenMLP(nn.Module):
    def __init__(self, seq_len: int = 1000):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(seq_len, 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Sigmoid(),
            nn.Linear(300, 2)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Oksuz(nn.Module):
    def __init__(self, gru: bool, input_size: int, hidden_size: int, seq_len: int = 1000,
                 num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        rnn = nn.GRU if gru else nn.LSTM
        self.rnn = rnn(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                       batch_first=True, bidirectional=bidirectional)
        self.fc1 = nn.Linear(in_features=((2 if bidirectional else 1) * hidden_size * seq_len) // input_size,
                             out_features=2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.rnn.input_size)
        rnn_out, *_ = self.rnn(x)
        rnn_out = rnn_out.reshape(batch_size, -1)
        fc_out = self.fc1(rnn_out)
        return fc_out


class Hoppe(nn.Module):
    def __init__(self, gru: bool, input_size: int, hidden_size: int, seq_len: int = 1000,
                 num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        rnn = nn.GRU if gru else nn.LSTM
        self.rnn = rnn(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                       batch_first=True, bidirectional=bidirectional)
        lstm_out_feature_size = ((2 if bidirectional else 1) * seq_len * hidden_size) // input_size
        fc1_out_feature_size = int(lstm_out_feature_size // 4.5)
        fc2_out_feature_size = int(fc1_out_feature_size // 1.5)
        fc3_out_feature_size = int(fc2_out_feature_size // 2)
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(num_features=lstm_out_feature_size),
            nn.Linear(in_features=lstm_out_feature_size, out_features=fc1_out_feature_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=fc1_out_feature_size),
            nn.Linear(in_features=fc1_out_feature_size, out_features=fc2_out_feature_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=fc2_out_feature_size),
            nn.Linear(in_features=fc2_out_feature_size, out_features=fc3_out_feature_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=fc3_out_feature_size),
            nn.Linear(in_features=fc3_out_feature_size, out_features=2),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.rnn.input_size)
        x, *_ = self.rnn(x)
        x = x.reshape(batch_size, -1)
        x = self.layers(x)
        return x


class Song(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(out_channels=16, kernel_size=21, stride=1, padding='same')
        nn.init.kaiming_normal(self.conv1d.weight, mode='fan_in')

    def forward(self, x):
        ...

    def resnet_layer(self):
        ...