import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pixelwise_dataset import PixelwiseDataset


class CohenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(1000, 300)
        self.fc1 = nn.Linear(300, 300)
        self.fc2 = nn.Linear(300, 300)
        self.output = nn.Linear(300, 2)

    def forward(self, fp):
        x1 = self.input(fp)
        x1 = torch.tanh(x1)
        x1 = self.fc1(x1)
        x1 = torch.tanh(x1)
        x1 = self.fc2(x1)
        x1 = self.output(x1)
        return x1


if __name__ == "__main__":
    dataset_loader = DataLoader(PixelwiseDataset.from_file_name("subj1_fisp_slc4_4"),
                                batch_size=1000,
                                shuffle=False,
                                pin_memory=True,
                                collate_fn=PixelwiseDataset.collate_fn)
    i = iter(dataset_loader)
    data, labels = next(i)
    data.cuda(), labels.cuda()
    print(CohenMLP().forward(data))


