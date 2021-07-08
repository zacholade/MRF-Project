from __future__ import annotations

import argparse
import os

import git
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

from config_parser import Configuration
from data_logger import DataLogger
from datasets import PixelwiseDataset, ScanwiseDataset
from networks import CohenMLP, OksuzLSTM
from transforms import NoiseTransform, ScaleLabels, ExcludeProtonDensity
from util import load_all_data_files


class TrainingAlgorithm:
    def __init__(self,
                 model,
                 optimiser: optim.Optimizer,
                 lr_scheduler,
                 initial_lr: float,
                 loss,
                 total_epochs: int,
                 batch_size: int,
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
        self.lr_scheduler = lr_scheduler
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

        self.base_directory = self._get_directory()
        self.logger = DataLogger(f"{self.base_directory}/Logs")

    def _get_directory(self):
        if not os.path.exists("Exports"):
            os.mkdir("Exports")

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        from datetime import datetime
        date = datetime.today().strftime('%Y-%m-%d_%H-%M')

        path = f"{self.model.__class__.__name__}_{date}_GIT-{sha}"
        path = f"DEBUG-{path}" if self.debug else path
        path = f"Exports/{path}"

        # This block of code makes sure the folder saving to is new and not been saved to before.
        if os.path.exists(path):
            num = 1
            while os.path.exists(f"{path}_{num}"):
                num += 1
            path = f"{path}_{num}"

        os.mkdir(path)
        return path

    def save(self, epoch):
        if not os.path.exists(f"{self.base_directory}/Models"):
            os.mkdir(f"{self.base_directory}/Models")

        filename = f"{self.model.__class__.__name__}_epoch-{epoch}_optim-{self.optimiser.__class__.__name__}_" \
                   f"initial-lr-{self.initial_lr}_loss-{self.loss.__class__.__name__}_batch-size-{self.batch_size}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'loss': self.loss,
            'batch_size': self.batch_size
        }, f"{self.base_directory}/Models/{filename}")

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)

        model = CohenMLP()
        model.load_state_dict(checkpoint['model_state_dict'])

        optimiser = optim.Adam(model.parameters())
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

        lr_scheduler = optim.lr_scheduler.StepLR  # TODO

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        batch_size = checkpoint['batch_size']

        number_epochs_to_do = epoch + 1
        return cls(model, optimiser, loss, number_epochs_to_do, batch_size,
                   starting_epoch=epoch)

    def train(self, data, labels, pos):
        self.model.train()
        predicted = self.model.forward(data)
        loss = self.loss(predicted, labels)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return predicted, loss

    def validate(self, data, labels, pos):
        self.model.eval()
        predicted = self.model.forward(data)
        loss = self.loss(predicted, labels)
        if self.debug:
            pass
            # plot(predicted, labels, pos)
        return predicted, loss

    def _should_break_early(self, current_iteration) -> bool:
        return self.debug and self.limit_iterations > 0 and \
               current_iteration % self.limit_iterations == 0 and \
               current_iteration != 0

    def loop(self, skip_valid):
        validation_transforms = transforms.Compose([ExcludeProtonDensity(), ScaleLabels(1000)])
        training_transforms = transforms.Compose([ExcludeProtonDensity(), ScaleLabels(1000), NoiseTransform(0, 0.01)])

        train_data, train_labels, train_file_lens = load_all_data_files("Train", file_limit=self.limit_number_files)
        valid_data, valid_labels, valid_file_lens = load_all_data_files("Test", file_limit=self.limit_number_files)
        training_dataset = PixelwiseDataset(train_data, train_labels, train_file_lens, transform=training_transforms)
        validation_dataset = ScanwiseDataset(valid_data, valid_labels, valid_file_lens, transform=validation_transforms)

        for epoch in range(self.starting_epoch + 1, self.total_epochs + 1):
            train_loader = DataLoader(training_dataset, pin_memory=True, collate_fn=PixelwiseDataset.collate_fn,
                                      num_workers=self.num_training_dataloader_workers,
                                      sampler=BatchSampler(RandomSampler(training_dataset),
                                                           batch_size=self.batch_size, drop_last=True))
            train_set = iter(train_loader)
            for current_iteration, (data, labels, pos) in enumerate(train_set):
                data, labels, pos = data.to(self.device), labels.to(self.device), pos.to(self.device)
                data = data.reshape((*data.shape, 1))
                if self._should_break_early(current_iteration):
                    break  # If in debug mode and we dont want to run the full epoch. Break early.

                current_iteration += 1
                if current_iteration % 100 == 0:
                    print(f"Epoch: {epoch}, Training iteration: {current_iteration} / "
                          f"{self.limit_iterations if self.debug else int(np.floor(len(training_dataset) / self.batch_size))}, "
                          f"LR: {self.lr_scheduler.get_last_lr()[0]}")
                predicted, loss = self.train(data, labels, pos)
                self.logger.log_error(predicted.detach(), labels.detach(), loss.detach(), data_type="train")

            if not skip_valid:
                print(f"Done training. Starting validation for epoch {epoch}.")
                validate_loader = torch.utils.data.DataLoader(validation_dataset,
                                                              batch_size=1,
                                                              collate_fn=ScanwiseDataset.collate_fn,
                                                              shuffle=False,
                                                              pin_memory=True,
                                                              num_workers=self.num_testing_dataloader_workers)
                validate_set = iter(validate_loader)

                for current_iteration, (data, labels, pos) in enumerate(validate_set):
                    print(f"Epoch: {epoch}, Validation scan: {current_iteration + 1} / "
                          f"{len(validate_loader)}")
                    data, labels, pos = data.to(self.device), labels.to(self.device), pos.to(self.device)
                    predicted, loss = self.validate(data, labels, pos)
                    self.logger.log_error(predicted.detach(), labels.detach(), loss.detach(), "valid")

            self.lr_scheduler.step()
            self.logger.log('learning_rate', self.lr_scheduler.get_last_lr())
            self.save(epoch)
            self.logger.on_epoch_end(epoch)
            print(f"Epoch {epoch} complete")


def main():
    parser = argparse.ArgumentParser()
    network_choices = ['cohen', 'oksuz_lstm']
    parser.add_argument('-network', choices=network_choices, type=str.lower, required=True)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-workers', '-num_workers', dest='num_workers', default=0, type=int)
    parser.add_argument('-skip_valid', '-no_valid', dest='skip_valid', action='store_true', default=False)
    args = parser.parse_args()

    config = Configuration(args.network, "config.ini", args.debug)

    print(f"Using {args.network} model.")
    print(f"Debug mode is {'enabled' if args.debug else 'disabled'}.")
    print(f"There are {args.num_workers} sub-process workers loading training data.")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}.")

    repo = git.Repo(search_parent_directories=True)
    if not config.debug and repo.is_dirty(submodules=False):
        print("Git head is not clean. Exiting...")
        import sys
        sys.exit(0)

    if args.network == 'cohen':
        model = CohenMLP()
    elif args.network == 'oksuz_lstm':
        model = OksuzLSTM()
    else:
        import sys
        print("Invalid model. Exiting...")
        sys.exit(1)

    optimiser = optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=config.lr_step_size, gamma=config.lr_gamma)
    loss = nn.MSELoss()
    trainer = TrainingAlgorithm(model,
                                optimiser,
                                lr_scheduler,
                                config.lr,
                                loss,
                                config.total_epochs,
                                config.batch_size,
                                num_training_dataloader_workers=args.num_workers,
                                num_testing_dataloader_workers=1,
                                debug=args.debug,
                                limit_number_files=config.limit_number_files,
                                limit_iterations=config.limit_iterations)
    trainer.loop(args.skip_valid)


if __name__ == "__main__":
    main()
