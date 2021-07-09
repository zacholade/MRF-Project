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
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(in_features=hidden_size, out_features=2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(dim=2)
        x = x.reshape(batch_size, -1, self.rnn.input_size)

        lstm_out, (hn, _) = self.rnn(x)
        hn = hn.squeeze()
        fc_out = self.fc(hn.squeeze())
        return fc_out
