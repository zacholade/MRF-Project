import numpy as np
import torch
from torch import nn as nn


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


class OksuzLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, seq_len: int = 1000,
                 num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=bidirectional)
        self.fc1 = nn.Linear(in_features=(2 if bidirectional else 1) * hidden_size * seq_len, out_features=2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.rnn.input_size)
        lstm_out, (hn, _) = self.rnn(x)
        lstm_out = lstm_out.reshape(batch_size, -1)
        fc_out = self.fc1(lstm_out)
        return fc_out

    #
    # def forward(self, x):
    #     # for i in range(100):
    #     #     x_ = np.arange(1000)
    #     #     print(x.shape)
    #     #     y = x[i].detach().cpu().numpy()
    #     #     import matplotlib.pyplot as plt
    #     #     plt.scatter(x_, y, s=1)
    #     #     plt.show()
    #
    #     batch_size = x.shape[0]
    #     # x = x.unsqueeze(dim=2)
    #     x = x.view(batch_size, -1, self.rnn.input_size)
    #     lstm_out, (hn, _) = self.rnn(x)
    #     # lstm_out = lstm_out.view(batch_size, -1)
    #     # print(lstm_out.shape)
    #     fc_out = self.fc(lstm_out[:, -1, :])
    #     return fc_out