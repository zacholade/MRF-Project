import os

import matplotlib.pyplot as plt
import numpy as np
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


def get_all_data_files(folder: str = "Train", *args, **kwargs):
    fingerprint_path = f"Data/MRF_maps/ExactFingerprintMaps/{folder}/"
    parameter_path = f"Data/MRF_maps/ParameterMaps/{folder}/"
    f1 = sorted([file.split('.')[0] for file in os.listdir(fingerprint_path) if not file.startswith(".")])
    f2 = sorted([file.split('.')[0] for file in os.listdir(parameter_path) if not file.startswith(".")])
    if f1 != f2:
        raise RuntimeError("Differing data inside Test/Train folders!")

    for file_name in f1:
        yield folder, file_name

def plot_model(network, testing_data_files):
    folder, file_name = next(testing_data_files)
    dataset = PixelwiseDataset.from_file_name(folder, file_name)
    predicted = network.forward(torch.FloatTensor(dataset.data).to("cuda")).to("cpu").detach().numpy()
    predicted_t1, predicted_t2 = np.transpose(predicted, axes=(2, 0, 1))
    actual_t1, actual_t2 = np.transpose(dataset.labels, axes=(2, 0, 1))

    plt.matshow(predicted_t1)
    plt.matshow(actual_t1)
    plt.matshow(np.abs(actual_t1 - predicted_t1))

    plt.matshow(predicted_t2)
    plt.matshow(actual_t2)
    plt.matshow(np.abs(actual_t2 - predicted_t2))

    plt.show()


if __name__ == "__main__":
    network = CohenMLP()
    trainer = TrainingAlgorithm(network, 0.001)
    testing_data_files = get_all_data_files("Test")
    for i, (folder, file_name) in enumerate(get_all_data_files("Train")):
        if i % 5 == 0 and i != 0:
            plot_model(network, testing_data_files)
        print(folder, file_name)
        dataset_loader = DataLoader(PixelwiseDataset.from_file_name(folder, file_name),
                                    batch_size=1000,
                                    shuffle=True,
                                    pin_memory=True,
                                    collate_fn=PixelwiseDataset.collate_fn)
        dataset_iter = iter(dataset_loader)
        try:
            data, labels = next(dataset_iter)
            while True:
                predicted = trainer.update(data, labels).to("cpu")
                print(torch.abs((predicted - labels).mean()))
                data, labels = next(dataset_iter)
        except StopIteration:
            continue
