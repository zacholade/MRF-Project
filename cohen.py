import torch
import torch.nn as nn
import torch.optim as optim
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


class TrainingAlgorithm:
    def __init__(self, network, lr):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.network = network.to(self.device)
        self.loss = nn.MSELoss()
        self.optimiser = optim.Adam(self.network.parameters(), lr=lr)

    def update(self, data, labels):
        self.network.train()
        data, labels = data.to(self.device), labels.to(self.device)
        predicted = self.network.forward(data)
        loss = self.loss(predicted, labels)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return predicted



    def save_model(self, filename):
        torch.save(self.network.state_dict(), f"models/{filename}.pth")

if __name__ == "__main__":
    dataset_loader = DataLoader(PixelwiseDataset.from_file_name("Train", "subj1_fisp_slc4_4"),
                                batch_size=1000,
                                shuffle=True,
                                pin_memory=True,
                                collate_fn=PixelwiseDataset.collate_fn)
    network = CohenMLP()
    trainer = TrainingAlgorithm(network, 0.05)
    dataset_iter = iter(dataset_loader)
    for i in range(52000):
        data, labels = next(dataset_iter)
        predicted = trainer.update(data, labels).to("cpu")
        print(torch.abs((predicted - labels).mean()))






