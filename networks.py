from torch import nn as nn


class CohenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1000, 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Sigmoid(),
            nn.Linear(300, 2)
        )

    def forward(self, x):
        return self.layers(x)


class OksuzLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.LSTM(input_size=1, hidden_size=1, batch_first=True)
        self.sig = nn.Sigmoid()
        self.l1 = nn.Linear(1000, 2)

    def forward(self, x):
        x, (hn, cn) = self.rnn1(x)
        x = self.sig(x.reshape(x.shape[0], x.shape[1]))
        x = self.l1(x)
        return x
