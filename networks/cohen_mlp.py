from torch import nn


class CohenMLP(nn.Module):
    """
    https://doi.org/10.1002/mrm.27198
    """
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