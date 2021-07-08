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
        self.rnn = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=50, out_features=2)

    def forward(self, x):
        lstm_out, (hn, _) = self.rnn(x)
        fc_out = self.fc(hn.squeeze())
        return fc_out
