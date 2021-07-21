from torch import nn


class RNNAttention(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.attention = CBAM(seq_len, no_spatial=True, pool_types=["max"])
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
        print(x.shape)
        x = x.unsqueeze(2)
        print(x.shape)
        x = self.attention(x)
        x = self.layers(x)
        return x