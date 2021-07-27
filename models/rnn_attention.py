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
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super().__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        # attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


class GeneralDotProductAttention(nn.Module):
    """
    https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weights = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)

    def forward(self, query, keys, values):
        query = query.unsqueeze(1)
        keys = keys.permute(0, 2, 1)  # B x seq_len x hidden -> B x hidden x seq_len
        scores = torch.bmm(query, keys)  # B x 1 x seq_len
        weights = torch.softmax(self.weights(scores.squeeze(1)).unsqueeze(1), dim=2)  # B x 1 x seq_len
        output = torch.bmm(weights, values).squeeze(1)  # B x hidden
        weights = weights.squeeze(1)  # B x seq_len
        return output, weights


class ScaledDotProductAttention(nn.Module):
    """
    https://arxiv.org/pdf/1409.0473.pdf
    Adapted for Many to one. Decoder is a linear layer.
    """
    def __init__(self, query_dim: int):
        super().__init__()
        self.scale = 1. / np.sqrt(query_dim)

    def forward(self, query, keys, values):
        query = query.unsqueeze(1)
        keys = keys.permute(0, 2, 1)  # B x seq_len x hidden -> B x hidden x seq_len
        scores = torch.bmm(query, keys)  # B x 1 x seq_len
        scores = torch.softmax(scores.mul_(self.scale), dim=2)  # B x 1 x seq_len
        output = torch.bmm(scores, values).squeeze(1)  # B x hidden
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
        self.attention = Attention(linear_dim, "general")

    def forward(self, x):
        batch_size = x.shape[0]
        fp = x[0, :]
        # x = self.attention(x, x, x)
        x = x.view(batch_size, -1, self.encoder.input_size)
        encoder_output, h_n = self.encoder(x)
        h_n = h_n.unsqueeze(1)
        attention_out, attention_weights = self.attention(encoder_output, h_n)
        labels = self.decoder(attention_out.view(batch_size, -1))
        # if not self.training:
        #     plot(plot_fp, attention_out[0, :, 0].detach().cpu().numpy(), 1)
        #     plot(plot_fp, fp.detach().cpu().numpy(), 1)
        return labels, attention_weights.squeeze(2)
