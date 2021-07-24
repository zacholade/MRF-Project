import torch
from torch import nn
import numpy as np

from util import plot, plot_fp


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        output, hidden = self.rnn(x)
        hidden = hidden[1] if isinstance(hidden, tuple) else hidden
        hidden = torch.cat([hidden[-1], hidden[-2]], dim=1) if \
            self.rnn.bidirectional else hidden[-1]

        return output, hidden


class Attention(nn.Module):
    """
    https://arxiv.org/pdf/1409.0473.pdf
    Adapted for Many to one. Decoder is a linear layer.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weights = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)

    def forward(self, x, h_n):
        batch_size, seq_len, _ = x.shape
        print(h_n.shape)
        print(h_n[:, -1])
        h = h_n.repeat(1, seq_len, 1)
        print(h.shape)
        print('attention forward')
        weights = self.weights(x)
        print(weights.shape)
        print(h_n.shape)
        score = torch.bmm(weights, h_n.unsqueeze(2)).squeeze(2)
        attention = torch.softmax(score.squeeze(2), dim=1)
        print(x.shape)
        print(attention.unsqueeze(2).shape)
        context = torch.bmm(x, attention.unsqueeze(2)).squeeze(2)
        print(context.shape)
        return context, attention


class ScaledDotProductAttention(nn.Module):
    """
    https://arxiv.org/pdf/1409.0473.pdf
    Adapted for Many to one. Decoder is a linear layer.
    """
    def __init__(self, query_dim: int):
        super().__init__()
        self.scale = 1. / np.sqrt(query_dim)

    def forward(self, query, keys, values):
        fps = query
        query = query.unsqueeze(2)
        keys = keys.unsqueeze(2).permute(0, 2, 1)  # B x seq_len x hidden -> B x hidden x seq_len
        scores = torch.bmm(query, keys)  # B x 1 x seq_len
        scores = torch.softmax(scores, dim=2)  # B x 1 x seq_len
        values = values.unsqueeze(2)
        output = torch.bmm(scores, values).squeeze(2)  # B x hidden
        scores = scores.squeeze(1)  # B x seq_len
        return output, scores


class RNNAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,  batch_size: int, seq_len: int = 1000,
                 num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, bidirectional=bidirectional)
        linear_dim = 2 * hidden_size if bidirectional else hidden_size
        lstm_out_feature_size = ((2 if bidirectional else 1) * seq_len * hidden_size) // input_size
        self.decoder = nn.Linear(lstm_out_feature_size, 2)
        self.attention = ScaledDotProductAttention(linear_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        # x = self.attention(x, x, x)
        x = x.view(batch_size, -1, self.encoder.input_size)
        encoder_output, h_n = self.encoder(x)
        attention_out = self.attention(h_n, encoder_output, encoder_output)
        labels = self.decoder(encoder_output)
        return labels
