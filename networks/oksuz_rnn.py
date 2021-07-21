from torch import nn


class OksuzRNN(nn.Module):
    """
    https://ieeexplore.ieee.org/document/8759502
    """
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