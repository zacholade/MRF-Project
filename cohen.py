from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

from datasets import PixelwiseDataset, ScanwiseDataset
from transforms import NoiseTransform, ScaleLabels, ExcludeProtonDensity
from util import get_all_data_files, load_all_data_files


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
    def __init__(self,
                 model,
                 optimiser: torch.optim.Optimizer,
                 loss: Callable,
                 total_epochs: int,
                 batch_size: int,
                 stats: ModelStats,
                 starting_epoch: int = 0,
                 limit_number_data: int = -1,
                 limit_iterations: int = None
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.optimiser = optimiser
        self.loss = loss

        self.starting_epoch = starting_epoch
        self.total_epochs = total_epochs

        self.batch_size = batch_size

        self.limit_number_data = limit_number_data
        self.limit_iterations = limit_iterations

        self.stats = stats

    def save(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'loss': self.loss,

        }, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)

        model = CohenMLP()
        model.load_state_dict(checkpoint['model_state_dict'])

        optimiser = optim.Adam(model.parameters())
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return cls(model, optimiser)

    def train(self, data, labels, pos):
        self.model.train()
        data, labels = data.to(self.device), labels.to(self.device)
        predicted = self.model.forward(data)
        loss = self.loss(predicted, labels)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return predicted, loss

    def validate(self, data, labels, pos):
        self.model.eval()
        data, labels = data.to(self.device), labels.to(self.device)
        predicted = self.model.forward(data)
        print(labels.shape)
        print(predicted.shape)
        loss = self.loss(predicted, labels)
        plot_model(predicted, labels, pos)
        return predicted, loss

    def save_model(self, filename):
        import git
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        torch.save(self.model.state_dict(), f"models/{filename}.pth")

    def _should_stop(self, current_iteration: int) -> bool:
        """
        Used to work out if the loop should break early based on the limit_iterations value.
        """
        return all([self.limit_iterations is not None,
                    self.limit_iterations > 0,
                    current_iteration % self.limit_iterations == 0,
                    current_iteration != 0])

    def loop(self, validate):
        validation_transforms = transforms.Compose([ExcludeProtonDensity(), ScaleLabels(1000)])
        training_transforms = transforms.Compose([ExcludeProtonDensity(), ScaleLabels(1000), NoiseTransform(0, 0.01)])

        train_data, train_labels = load_all_data_files("Train", file_limit=self.limit_number_data)
        valid_data, valid_labels = load_all_data_files("Test", file_limit=self.limit_number_data)
        training_dataset = PixelwiseDataset(train_data, train_labels, transform=training_transforms)
        validation_dataset = ScanwiseDataset(valid_data, valid_labels, transform=validation_transforms)

        print(f"There will be approx {min(len(training_dataset) / self.batch_size, self.limit_iterations)} iterations per epoch.")
        for epoch in range(self.starting_epoch + 1, self.total_epochs + 1):
            train_loader = torch.utils.data.DataLoader(training_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       collate_fn=PixelwiseDataset.collate_fn,
                                                       num_workers=1,
                                                       drop_last=True)

            for current_iteration, (data, labels, pos) in enumerate(train_loader):
                if self._should_stop(current_iteration):
                    break

                current_iteration += 1

                predicted, loss = self.train(data, labels, pos)
                print(f"Epoch: {epoch}, Training iteration: {current_iteration} / "
                      f"≈{min(len(training_dataset) / self.batch_size, self.limit_iterations)}")
                self.stats.update(predicted.cpu(), labels.cpu())

            if not validate:
                continue

            print(f"Done training. Starting validation for epoch {epoch}.")
            # Eval
            validate_loader = torch.utils.data.DataLoader(validation_dataset,
                                                          batch_size=1,
                                                          shuffle=False,
                                                          pin_memory=True,
                                                          num_workers=1,
                                                          collate_fn=ScanwiseDataset.collate_fn)
            data, labels, pos = next(iter(validate_loader))
            predicted, loss = self.validate(data, labels, pos)

            print(f"Epoch {epoch} complete")
            #  MEAN ABSOLUTE PERCENTAGE ERROR!!
            self.save(epoch, "model")


def plot_model(predicted, labels, pos):
    print("Plotting")
    predicted_t1, predicted_t2 = predicted.cpu().detach().numpy().transpose()
    actual_t1, actual_t2 = labels.cpu().numpy().transpose()

    x = (pos // 230).cpu().numpy().astype(int)
    y = (pos % 230).cpu().numpy().astype(int)

    predicted_t1_map, predicted_t2_map = np.zeros((230, 230)), np.zeros((230, 230))
    actual_t1_map, actual_t2_map = np.zeros((230, 230)), np.zeros((230, 230))
    predicted_t1_map[x, y] = predicted_t1
    predicted_t2_map[x, y] = predicted_t2
    actual_t1_map[x, y] = actual_t1
    actual_t2_map[x, y] = actual_t2

    plt.matshow(predicted_t1_map, norm=plt.Normalize())
    plt.title("Predicted T1")
    plt.matshow(actual_t1_map, norm=plt.Normalize())
    plt.title("Actual T1")
    plt.matshow(np.abs(actual_t1_map - predicted_t1_map), norm=plt.Normalize())
    plt.title("abs(predicted - actual) T1")

    plt.matshow(predicted_t2_map, norm=plt.Normalize())
    plt.title("Predicted T2")
    plt.matshow(actual_t2_map, norm=plt.Normalize())
    plt.title("Actual T2")
    plt.matshow(np.abs(actual_t2_map - predicted_t2_map), norm=plt.Normalize())
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


def main():
    total_epochs = 100
    batch_size = 3000
    learning_rate = 0.001
    validate = True

    # For debug only.
    limit_iterations = 500  # Set to 0 to not limit.
    limit_number_data: int = 15

    if limit_number_data > -1 or limit_iterations > 0:
        for _ in range(5):
            print("WARNING!!!!!! NOT USING FULL DATASET OR LIMITING ITERATIONS.")

    model = CohenMLP()
    stats = ModelStats()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.MSELoss()
    trainer = TrainingAlgorithm(model,
                                optimiser,
                                loss,
                                total_epochs,
                                batch_size,
                                stats,
                                limit_number_data=limit_number_data,
                                limit_iterations=limit_iterations)
    trainer.loop(validate)


if __name__ == "__main__":
    main()
