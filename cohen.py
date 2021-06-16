from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

from pixelwise_dataset import PixelwiseDataset
from transforms import NoiseTransform, ScaleLabels, ExcludeProtonDensity


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

    def get_checkpoint(self):
        return {"model_state_dict": self.state_dict()}


class TrainingAlgorithm:
    def __init__(self, network, lr):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.network = network.to(self.device)
        self.loss = nn.MSELoss()
        self.optimiser = optim.Adam(self.network.parameters(), lr=lr)

    def train(self, data, labels):
        self.network.train()
        data, labels = data.to(self.device), labels.to(self.device)
        predicted = self.network.forward(data)
        loss = self.loss(predicted, labels)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return predicted, loss

    def validate(self, data, labels):
        self.network.eval()
        data, labels = data.to(self.device), labels.to(self.device)
        predicted = self.network.forward(data)
        loss = self.loss(predicted, labels)
        plot_model(predicted, labels)
        return predicted, loss

    def save_model(self, filename):
        torch.save(self.network.state_dict(), f"models/{filename}.pth")

    # def loop(self):
    #     for epoch in range (self.current_epoch + 1, self.total_epochs + 1):
    #


def plot_model(predicted, labels):
    print("Plotting")
    predicted = predicted.to("cpu").detach().numpy()
    predicted_t1, predicted_t2 = np.transpose(predicted).reshape(2, 230, 230)
    actual_t1, actual_t2 = np.transpose(labels.cpu().numpy()).reshape(2, 230, 230)

    plt.matshow(predicted_t1, norm=plt.Normalize())
    plt.title("Predicted T1")
    plt.matshow(actual_t1, norm=plt.Normalize())
    plt.title("Actual T1")
    plt.matshow(np.abs(actual_t1 - predicted_t1), norm=plt.Normalize())
    plt.title("abs(predicted - actual) T1")

    plt.matshow(predicted_t2, norm=plt.Normalize())
    plt.title("Predicted T2")
    plt.matshow(actual_t2, norm=plt.Normalize())
    plt.title("Actual T2")
    plt.matshow(np.abs(actual_t2 - predicted_t2), norm=plt.Normalize())
    plt.title("abs(predicted - actual) T2")

    plt.show()


class ModelStats:
    def __init__(self):
        self.epoch = 0
        self.loss_func = nn.MSELoss()

    def step_epoch(self):
        self.epoch += 1

    def update(self, y_pred, y_true):
        acc = (y_true / y_pred).mean()
        rmse_loss = torch.sqrt(self.loss_func(y_pred, y_true))
        t1_pred, t2_pred = torch.transpose(y_pred, 0, 1)
        t1_true, t2_true = torch.transpose(y_true, 0, 1)
        t1_rmse_loss = torch.sqrt(self.loss_func(t1_pred, t1_true))
        t2_rmse_loss = torch.sqrt(self.loss_func(t2_pred, t2_true))
        print(f"RMSE: {rmse_loss.item()}, T1 RMSE: {t1_rmse_loss.item()}, T2 RMSE: {t2_rmse_loss.item()}")


if __name__ == "__main__":
    epochs = 100
    batch_size = 10000
    learning_rate = 0.001
    validate = True
    limit_iterations = 300  # Set to 0 to not limit.

    model = CohenMLP()
    trainer = TrainingAlgorithm(model, learning_rate)
    stats = ModelStats()

    validation_transforms = transforms.Compose([ExcludeProtonDensity(), ScaleLabels(1000)])
    training_transforms = transforms.Compose([ExcludeProtonDensity(), ScaleLabels(1000), NoiseTransform(0, 0.01)])

    training_dataset = PixelwiseDataset("Train", transform=training_transforms)
    validation_dataset = PixelwiseDataset("Test", transform=validation_transforms)

    print(f"There will be approx {min(len(training_dataset) / batch_size, limit_iterations)} iterations per epoch.")
    for epoch in range(1, epochs + 1):
        # Train
        train_loader = torch.utils.data.DataLoader(training_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   collate_fn=PixelwiseDataset.collate_fn,
                                                   num_workers=6,
                                                   worker_init_fn=training_dataset.worker_init_fn,
                                                   drop_last=True)
        for iteration_counter, (data, labels) in enumerate(train_loader):
            if limit_iterations > 0 and iteration_counter % limit_iterations == 0 and iteration_counter != 0:
                break

            iteration_counter += 1
            predicted, loss = trainer.train(data, labels)
            print(f"Epoch: {epoch}, Training iteration: {iteration_counter} / ≈{min(len(training_dataset) / batch_size, limit_iterations)}")
            stats.update(predicted.cpu(), labels.cpu())
        if not validate:
            continue

        print("Done training. Starting validation.")
        # Eval
        validate_loader = torch.utils.data.DataLoader(validation_dataset,
                                                      batch_size=validation_dataset.num_pixels_per_matrix,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      collate_fn=PixelwiseDataset.collate_fn,
                                                      worker_init_fn=validation_dataset.worker_init_fn,
                                                      num_workers=1)
        data, labels = next(iter(validate_loader))
        predicted, loss = trainer.validate(data, labels)

        print(f"Epoch {epoch} complete")

