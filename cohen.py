from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

from config_parser import Configuration
from datasets import PixelwiseDataset, ScanwiseDataset
from transforms import NoiseTransform, ScaleLabels, ExcludeProtonDensity
from util import load_all_data_files

import git
import os
import argparse


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


class TrainingAlgorithm:
    def __init__(self,
                 model,
                 optimiser: torch.optim.Optimizer,
                 initial_lr: float,
                 loss,
                 total_epochs: int,
                 batch_size: int,
                 stats: ModelStats,
                 starting_epoch: int = 0,
                 num_training_dataloader_workers: int = 1,
                 num_testing_dataloader_workers: int = 1,
                 debug: bool = False,
                 limit_number_files: int = -1,
                 limit_iterations: int = -1
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.optimiser = optimiser
        self.initial_lr = initial_lr
        self.loss = loss
        self.starting_epoch = starting_epoch
        self.total_epochs = total_epochs
        self.batch_size = batch_size

        self.num_training_dataloader_workers = num_training_dataloader_workers
        self.num_testing_dataloader_workers = num_testing_dataloader_workers

        self.debug = debug
        self.limit_number_files = limit_number_files
        self.limit_iterations = limit_iterations

        self.stats = stats

        self.directory = self._get_directory()

    def _get_directory(self):
        if not os.path.exists("Exports"):
            os.mkdir("Exports")

        if not os.path.exists("Exports"):
            os.mkdir("Exports")

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        from datetime import datetime
        date = datetime.today().strftime('%Y-%m-%d_%H-%M')

        path = f"Exports/{self.model.__class__.__name__}_{date}_GIT-{sha}"

        # This block of code makes sure the folder saving to is new and not been saved to before.
        if os.path.exists(path):
            num = 1
            while os.path.exists(f"{path}_{num}"):
                num += 1
            path = f"{path}_{num}"
        return path

    def save(self, epoch):
        filename = f"cohen_epoch-{epoch}_optim-{self.optimiser.__class__.__name__}_" \
                   f"initial-lr-{self.initial_lr}_loss-{self.loss.__class__.__name__}_batch-size-{self.batch_size}"

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'loss': self.loss,
            'batch_size': self.batch_size
        }, f"{self.directory}/Exports/{filename}")

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)

        model = CohenMLP()
        model.load_state_dict(checkpoint['model_state_dict'])

        optimiser = optim.Adam(model.parameters())
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        batch_size = checkpoint['batch_size']
        stats = ModelStats()

        number_epochs_to_do = epoch + 1
        return cls(model, optimiser, loss, number_epochs_to_do, batch_size,
                   starting_epoch=epoch, stats=stats)

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
        loss = self.loss(predicted, labels)
        if self.debug:
            plot_model(predicted, labels, pos)
        return predicted, loss

    def loop(self, validate):
        validation_transforms = transforms.Compose([ExcludeProtonDensity(), ScaleLabels(1000)])
        training_transforms = transforms.Compose([ExcludeProtonDensity(), ScaleLabels(1000), NoiseTransform(0, 0.01)])

        train_data, train_labels = load_all_data_files("Train", file_limit=self.limit_number_files)
        valid_data, valid_labels = load_all_data_files("Test", file_limit=self.limit_number_files)
        training_dataset = PixelwiseDataset(train_data, train_labels, transform=training_transforms)
        validation_dataset = ScanwiseDataset(valid_data, valid_labels, transform=validation_transforms)

        for epoch in range(self.starting_epoch + 1, self.total_epochs + 1):
            train_loader = torch.utils.data.DataLoader(training_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       collate_fn=PixelwiseDataset.collate_fn,
                                                       num_workers=self.num_training_dataloader_workers,
                                                       drop_last=True)

            for current_iteration, (data, labels, pos) in enumerate(train_loader):
                if all([self.debug, self.limit_iterations > 0,
                        current_iteration % self.limit_iterations == 0,
                        current_iteration != 0]):
                    break  # If in debug mode and we dont want to run the full epoch. Break early.

                current_iteration += 1

                predicted, loss = self.train(data, labels, pos)
                print(f"Epoch: {epoch}, Training iteration: {current_iteration} / "
                      f"≈{self.limit_iterations if self.debug else len(training_dataset) / self.batch_size}")
                self.stats.update(predicted.cpu(), labels.cpu())

            if not validate:
                continue

            print(f"Done training. Starting validation for epoch {epoch}.")
            # Eval
            validate_loader = torch.utils.data.DataLoader(validation_dataset,
                                                          batch_size=1,
                                                          shuffle=False,
                                                          pin_memory=True,
                                                          num_workers=self.num_testing_dataloader_workers,
                                                          collate_fn=ScanwiseDataset.collate_fn)
            data, labels, pos = next(iter(validate_loader))
            predicted, loss = self.validate(data, labels, pos)

            print(f"Epoch {epoch} complete")
            #  MEAN ABSOLUTE PERCENTAGE ERROR!!
            self.save(epoch)


def plot_model(predicted, labels, pos, save_dir: str = None):
    """
    :param predicted: The predicted t1 and t2 labels.
    :param labels: The ground-truth t1 and t2 labels.
    :param pos: The index position matrix for each t1 and t2 value.
    :param save_dir: Optional argument. Saves the plots to that directory if not None.
    """
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

    plt.matshow(predicted_t1_map)
    plt.title("Predicted T1")
    plt.clim(0, 3000)
    plt.colorbar(shrink=0.8, label='milliseconds')

    plt.matshow(actual_t1_map)
    plt.title("Actual T1")
    plt.clim(0, 3000)
    plt.colorbar(shrink=0.8, label='milliseconds')

    plt.matshow(np.abs(actual_t1_map - predicted_t1_map))
    plt.title("abs(predicted - actual) T1")
    plt.clim(0, 3000)
    plt.colorbar(shrink=0.8, label='milliseconds')

    plt.matshow(predicted_t2_map)
    plt.title("Predicted T2")
    plt.clim(0, 300)
    plt.colorbar(shrink=0.8, label='milliseconds')

    plt.matshow(actual_t2_map)
    plt.title("Actual T2")
    plt.clim(0, 300)
    plt.colorbar(shrink=0.8, label='milliseconds')

    plt.matshow(np.abs(actual_t2_map - predicted_t2_map))
    plt.title("abs(predicted - actual) T2")
    plt.clim(0, 300)
    plt.colorbar(shrink=0.8, label='milliseconds')

    plt.savefig('bop.png')
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-debug', action='store_true', default=False)
    args = parser.parse_args()
    config = Configuration("config.ini", args.debug)

    num_training_dataloader_workers = config.num_training_dataloader_workers if not config.debug else 1
    num_testing_dataloader_workers = config.num_testing_dataloader_workers if not config.debug else 1

    print(f"Debug mode is {'enabled' if args.debug else 'disabled'}.")

    repo = git.Repo(search_parent_directories=True)
    if not config.debug and repo.is_dirty(submodules=False):
        print("Git head is not clean. Exiting...")
        import sys
        sys.exit(0)

    model = CohenMLP()
    stats = ModelStats()
    optimiser = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss = nn.MSELoss()
    trainer = TrainingAlgorithm(model,
                                optimiser,
                                config.learning_rate,
                                loss,
                                config.total_epochs,
                                config.batch_size,
                                stats,
                                num_training_dataloader_workers=num_training_dataloader_workers,
                                num_testing_dataloader_workers=num_testing_dataloader_workers,
                                debug=args.debug,
                                limit_number_files=config.limit_number_files,
                                limit_iterations=config.limit_iterations)
    trainer.loop(config.validate)


if __name__ == "__main__":
    main()
