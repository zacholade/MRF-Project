import torch
from torch import nn
import torch.nn.functional as F

from networks.sub_layers import MedianPool2d


class Hoppe(nn.Module):
    """
    Hoppe, E., Thamm, F., Körzdörfer, G., Syben, C., Schirrmacher, F., Nittka, M., Pfeuffer, J.,
    Meyer, H. and Maier, A.K., 2019, September.
    Magnetic Resonance Fingerprinting Reconstruction Using Recurrent Neural Networks.
    In GMDS (pp. 126-133).
    """
    def __init__(self, gru: bool, input_size: int, hidden_size: int, seq_len: int = 1000,
                 num_layers: int = 1, bidirectional: bool = False, spatial_pooling: str = None):
        super().__init__()
        rnn = nn.GRU if gru else nn.LSTM
        self.rnn = rnn(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                       batch_first=True, bidirectional=bidirectional)
        lstm_out_feature_size = ((2 if bidirectional else 1) * seq_len * hidden_size) // input_size
        # Ensure network output ratios match Hoppe's network architecture for variable fingerprint dimensions.
        # Hoppe uses: 9000 -> 2000 -> 1333 -> 666 -> 2
        # Which are the ratios: 4.5 -> 1.5 -> 2 -> and last ratio is just going to 2 for t1/t2.
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
        if spatial_pooling is None:
            self.spatial_pooling = None
        elif spatial_pooling == "mean":
            self.spatial_pooling = nn.AvgPool2d((3, 3), (1, 1), padding=1, count_include_pad=False)
        elif spatial_pooling == "median":
            self.spatial_pooling = MedianPool2d(3, 1, padding=1)
        else:
            raise ValueError(f"Unknown pooling operation: {spatial_pooling}.")

    def forward(self, x, pos=None):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.rnn.input_size)
        x, *_ = self.rnn(x)
        x = x.reshape(batch_size, -1)
        x = self.layers(x)
        if self.spatial_pooling and not self.training:
            # Apply pooling operation. Reduction by 9 in the 2nd dimension. (3x3 patches).
            x_ = (pos // 230).type(torch.LongTensor)
            y_ = (pos % 230).type(torch.LongTensor)
            empty = torch.empty((1, 230, 230, 2), device='cuda' if torch.cuda.is_available() else 'cpu')
            empty[:, x_, y_] = x
            x = empty.transpose(3, 1)
            x = self.spatial_pooling(x).squeeze(0)
            x = x.transpose(2, 0)
            x = x[x_, y_]
        return x


# i = torch.arange(0, 50).reshape(1, 2, 5, 5).type(torch.FloatTensor)
# i[:, :, torch.LongTensor([2, 3, 4]), torch.LongTensor([2, 3, 4])] = 1
# # i = F.pad(i, (1, 1, 1, 1))
# print(i.squeeze(0))
# print(i.shape)
# pool = MedianPool2d(3, 1, padding=1)
# pooled = pool(i)
# print(pooled.shape)
# print(pooled)
