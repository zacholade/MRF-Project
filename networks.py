from torch import nn as nn


class CohenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(1000, 300)
        self.tanh1 = nn.Tanh()
        self.fc1 = nn.Linear(300, 300)
        self.tanh2 = nn.Tanh()
        self.fc2 = nn.Linear(300, 300)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(300, 2)

    def forward(self, fp):
        x1 = self.input(fp)
        x1 = self.tanh1(x1)
        x1 = self.fc1(x1)
        x1 = self.tanh2(x1)
        x1 = self.fc2(x1)
        x1 = self.sigmoid(x1)
        x1 = self.output(x1)
        return x1


class Rinq(nn.Module):
    ...