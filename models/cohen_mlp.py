from torch import nn


class CohenMLP(nn.Module):
    """
    https://doi.org/10.1002/mrm.27198
    """
    def __init__(self, seq_len: int = 1000, modern: bool = False):
        super().__init__()
        if modern:
            self.layers = nn.Sequential(
                nn.Linear(seq_len, 300),
                nn.BatchNorm1d(300),
                nn.ReLU(),
                nn.Linear(300, 300),
                nn.BatchNorm1d(300),
                nn.ReLU(),
                nn.Linear(300, 300),
                nn.BatchNorm1d(300),
                nn.ReLU(),
                nn.Linear(300, 2)
            )
        else:
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
        x = self.layers(x)
        return x