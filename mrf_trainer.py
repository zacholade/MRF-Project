from __future__ import annotations

import argparse
import logging
import os
import sys

import git
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

from config_parser import Configuration
from data_logger import DataLogger
from datasets import PixelwiseDataset, ScanwiseDataset, PatchwiseDataset, ScanPatchwiseDataset
from logging_manager import setup_logging, LoggingMixin
from models import (CohenMLP, OksuzRNN, Hoppe, RNNAttention, Song,
                    RCAUNet, PatchSizeTest, R2Plus1DCbam, R2Plus1DNonLocal,
                    Balsiger, R2Plus1DTemporalNonLocal, Soyak)
from models.r2plus1d import R2Plus1D
from transforms import NoiseTransform, OnlyT1T2, ApplyPD, Normalise, Unnormalise
from util import load_all_data_files, plot, get_exports_dir, plot_maps, plot_fp

np.set_printoptions(threshold=sys.maxsize)


class TrainingAlgorithm(LoggingMixin):
    def __init__(self,
                 model,
                 optimiser: optim.Optimizer,
                 lr_scheduler,
                 initial_lr: float,
                 loss,
                 total_epochs: int,
                 batch_size: int,
                 export_dir: str,
                 seq_len: int = 1000,
                 starting_epoch: int = 0,
                 num_training_dataloader_workers: int = 1,
                 num_testing_dataloader_workers: int = 1,
                 plot_every: int = -1,
                 using_attention: bool = False,
                 using_spatial: bool = False,
                 valid_chunks: int = 1,
                 debug: bool = False,
                 limit_number_files: int = -1,
                 limit_iterations: int = -1,
                 device: str = None,
    ):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimiser = optimiser
        self.lr_scheduler = lr_scheduler
        self.initial_lr = initial_lr
        self.loss = loss
        self.seq_len = seq_len
        self.starting_epoch = starting_epoch
        self.total_epochs = total_epochs
        self.batch_size = batch_size

        self.num_training_dataloader_workers = num_training_dataloader_workers
        self.num_testing_dataloader_workers = num_testing_dataloader_workers

        self.plot_every = plot_every  # Save a reconstruction plot every n epochs.
        self.using_attention = using_attention
        self.using_spatial = using_spatial
        self.valid_chunks = valid_chunks
        self.debug = debug
        self.limit_number_files = limit_number_files
        self.limit_iterations = limit_iterations

        self.export_dir = export_dir
        self.data_logger = DataLogger(f"{self.export_dir}/Logs")

        # Early stop logic
        self._lowest_error = np.inf
        self._patience = 15
        self._best_epoch = 0
        self._epochs_without_improvement = 0
        self._should_stop = False

    def save(self, epoch):
        if not os.path.exists(f"{self.export_dir}/Models"):
            os.mkdir(f"{self.export_dir}/Models")

        if not os.path.exists(f"{self.export_dir}/CompleteModels"):
            os.mkdir(f"{self.export_dir}/CompleteModels")

        filename = f"{self.model.__class__.__name__}_epoch-{epoch}_optim-{self.optimiser.__class__.__name__}_" \
                   f"initial-lr-{self.initial_lr}_loss-{self.loss.__class__.__name__}_batch-size-{self.batch_size}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'model': self.model,
            'optimiser_state_dict': self.optimiser.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'loss': self.loss,
            'batch_size': self.batch_size
        }, f"{self.export_dir}/Models/{filename}")

        torch.save(self.model, f"{self.export_dir}/CompleteModels/complete_{filename}")

    def train(self, data, labels):
        self.model.train()
        predicted = self.model.forward(data)

        attention = None
        if isinstance(predicted, tuple):
            attention = predicted[1]
            predicted = predicted[0]

        loss = self.loss(predicted, labels)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return predicted, loss, attention

    def validate(self, data, labels):
        self.model.eval()
        predicted = self.model.forward(data)

        attention = None
        if isinstance(predicted, tuple):
            attention = predicted[1]
            predicted = predicted[0]

        loss = self.loss(predicted, labels)
        return predicted, loss, attention

    def _should_break_early(self, current_iteration) -> bool:
        return self.debug and self.limit_iterations > 0 and \
               current_iteration % self.limit_iterations == 0 and \
               current_iteration != 0

    def valid_loop(self, epoch, validate_loader):
        """
        A validate loop. Because we can't just feed an entire scan into the model
        (27000+ batch size causes memory issues), we chunk the scan up into chunks and feed them one
        by one. Then we combine them all before plotting/calculating loss etc.
        """
        validate_set = iter(validate_loader)
        self.model.eval()
        epoch_mape = 0
        total_scans = 0
        current_chunk = 0  # The current chunk of one scan currently processing
        chunk_loss = 0
        chunk_predicted = None
        chunk_labels = None
        chunk_pos = None
        for current_iteration, (data, labels, pos, file_name) in enumerate(validate_set):
            current_chunk += 1
            self.logger.info(f"Epoch: {epoch}, Validation scan: {(current_iteration // self.valid_chunks) + 1} / "
                             f"{len(validate_loader) // self.valid_chunks}. "
                             f"Chunk: {((current_chunk - 1) % self.valid_chunks) + 1} / "
                             f"{self.valid_chunks}")
            data, labels = data.to(self.device), labels.to(self.device)
            predicted, loss, attention = self.validate(data, labels)
            data, labels = data.cpu(), labels.cpu()
            chunk_loss += loss.detach().cpu().item()
            chunk_predicted = predicted.detach().cpu() if chunk_predicted is None else \
                torch.cat((chunk_predicted, predicted.detach().cpu()), 0)
            chunk_labels = labels.detach().cpu() if chunk_labels is None else \
                torch.cat((chunk_labels, labels.detach().cpu()), 0)
            chunk_pos = pos.detach().cpu() if chunk_pos is None else \
                torch.cat((chunk_pos, pos.detach().cpu()), 0)
            if current_chunk % self.valid_chunks == 0:
                self.data_logger.log_error(chunk_predicted,
                                           chunk_labels,
                                           chunk_loss, "valid")

                if self.plot_every > 0 and epoch % self.plot_every == 0:
                    if not os.path.exists(f"{self.export_dir}/Plots"):
                        os.mkdir(f"{self.export_dir}/Plots")
                    # Matplotlib has a memory leak. To alleviate this do plotting in a subprocess and
                    # join to it. When process is suspended, memory is forcibly released.
                    plot(plot_maps,
                         chunk_predicted.numpy(),
                         chunk_labels.numpy(),
                         chunk_pos.numpy().astype(int),
                         epoch,
                         f"{self.export_dir}/Plots/{file_name}",
                         file_name)
                epoch_mape += torch.mean(torch.abs(((chunk_labels - chunk_predicted) / chunk_labels))) * 100
                total_scans += 1
                chunk_loss = 0
                chunk_predicted = None
                chunk_labels = None
                chunk_pos = None

        epoch_valid_mape = epoch_mape / total_scans
        self.logger.info(f"Validation MAPE: {epoch_valid_mape}")
        return epoch_valid_mape

    def loop(self, skip_valid):
        validation_transforms = transforms.Compose([Unnormalise(), ApplyPD(), Normalise(), NoiseTransform(0, 0.01), OnlyT1T2()])
        training_transforms = transforms.Compose([Unnormalise(), ApplyPD(), Normalise(), NoiseTransform(0, 0.01), OnlyT1T2()])

        if self.using_spatial:
            (train_data, train_labels, train_file_lens, train_file_names, train_pos), \
            (valid_data, valid_labels, valid_file_lens, valid_file_names, valid_pos) = \
                load_all_data_files(seq_len=self.seq_len,
                                    file_limit=self.limit_number_files,
                                    compressed=False,
                                    debug=self.debug)
            training_dataset = PatchwiseDataset(self.model.patch_size, train_pos, train_data, train_labels, train_file_lens,
                                                train_file_names, transform=training_transforms)
            validation_dataset = ScanPatchwiseDataset(self.valid_chunks, self.model.patch_size, valid_pos, valid_data, valid_labels, valid_file_lens,
                                                      valid_file_names, transform=validation_transforms)
        else:
            (train_data, train_labels, train_file_lens, train_file_names), \
            (valid_data, valid_labels, valid_file_lens, valid_file_names) = \
                load_all_data_files(seq_len=self.seq_len,
                                    file_limit=self.limit_number_files,
                                    compressed=True,
                                    debug=self.debug)
            training_dataset = PixelwiseDataset(train_data, train_labels, train_file_lens,
                                                train_file_names, transform=training_transforms)
            validation_dataset = ScanwiseDataset(self.valid_chunks, valid_data, valid_labels, valid_file_lens,
                                                 valid_file_names, transform=validation_transforms)

        total_iterations = self.limit_iterations if (self.debug and self.limit_iterations > 0) else \
            int(np.floor(len(training_dataset) / self.batch_size))

        for epoch in range(self.starting_epoch + 1, self.total_epochs + 1):
            for _ in range(2):
                train_loader = DataLoader(training_dataset, pin_memory=True, collate_fn=PixelwiseDataset.collate_fn,
                                          num_workers=self.num_training_dataloader_workers,
                                          sampler=BatchSampler(RandomSampler(training_dataset),
                                                               batch_size=self.batch_size, drop_last=True))
                train_set = iter(train_loader)
                for current_iteration, (data, labels, pos) in enumerate(train_set):
                    data, labels = data.to(self.device), labels.to(self.device)
                    if self._should_break_early(current_iteration):
                        break  # If in debug mode and we dont want to run the full epoch. Break early.

                    current_iteration += 1
                    predicted, loss, attention = self.train(data, labels)

                    data, labels = data.cpu(), labels.cpu()
                    self.data_logger.log_error(predicted.detach().cpu(),
                                               labels.detach().cpu(),
                                               loss.detach().cpu().item(),
                                               data_type="train")

                    if current_iteration % max((total_iterations // 150), 1) == 0 or current_iteration == 2:
                        self.logger.info(f"Epoch: {epoch}, Training iteration: {current_iteration} / "
                                         f"{self.limit_iterations if (self.debug and self.limit_iterations > 0) else int(np.floor(len(training_dataset) / self.batch_size))}, "
                                         f"LR: {self.lr_scheduler.get_last_lr()[0]}, "
                                         f"Loss: {loss}")
                        if attention is not None:
                            ...
                            # plot(plot_fp, attention[0].detach().cpu().numpy(), f"{epoch}_{current_iteration}", save_dir=self.export_dir)

            if not skip_valid:
                with torch.no_grad():
                    self.logger.info(f"Done training. Starting validation for epoch {epoch}.")
                    validate_loader = torch.utils.data.DataLoader(validation_dataset,
                                                                  batch_size=1,
                                                                  collate_fn=ScanwiseDataset.collate_fn,
                                                                  shuffle=False,
                                                                  pin_memory=True,
                                                                  num_workers=self.num_testing_dataloader_workers)
                    validation_mape = self.valid_loop(epoch, validate_loader)
            self.lr_scheduler.step()
            self.data_logger.log('learning_rate', self.lr_scheduler.get_last_lr()[0])
            self.save(epoch)
            self.data_logger.on_epoch_end(epoch)
            self.logger.info(f"Epoch {epoch} complete")

            if epoch > 50:
                # Early stop logic
                self._epochs_without_improvement += 1
                if validation_mape < self._lowest_error:
                    self._epochs_without_improvement = 0
                    self._best_epoch = epoch
                    self._lowest_error = validation_mape
                if self._epochs_without_improvement > self._patience:
                    self.logger.warning(f"{self._patience} epochs have passed without improving. Stopping training.")
                    self._should_stop = True  # Terminate training here if this reaches.
                    break

                self.logger.info(f"Epochs wt/out improv: {self._epochs_without_improvement}, "
                                 f"Best epoch: {self._best_epoch}, "
                                 f"Lowest Valid MAPE: {self._lowest_error}, "
                                 f"Should Stop: {self._should_stop}")


def get_network(network: str, config):
    using_spatial = False  # If true input is fed as patches.
    using_attention = False

    if network == 'cohen':
        model = CohenMLP(seq_len=config.seq_len)
    elif network == 'oksuz_rnn':
        model = OksuzRNN(config.gru, input_size=config.rnn_input_size, hidden_size=config.rnn_hidden_size,
                      seq_len=config.seq_len, num_layers=config.rnn_num_layers,
                      bidirectional=config.rnn_bidirectional)
    elif network == 'hoppe':
        spatial_pooling = None if config.spatial_pooling.lower() == 'none' else config.spatial_pooling.lower()
        using_spatial = True if spatial_pooling is not None else False
        model = Hoppe(config.gru, input_size=config.rnn_input_size, hidden_size=config.rnn_hidden_size,
                      seq_len=config.seq_len, num_layers=config.rnn_num_layers,
                      bidirectional=config.rnn_bidirectional, spatial_pooling=spatial_pooling,
                      patch_size=config.patch_size)
    elif network == 'rnn_attention':
        using_attention = True
        model = RNNAttention(input_size=config.rnn_input_size, hidden_size=config.rnn_hidden_size,
                             batch_size=config.batch_size, seq_len=config.seq_len,
                             num_layers=config.rnn_num_layers, bidirectional=config.rnn_bidirectional)
    elif network == 'song':
        using_attention=True
        model = Song(seq_len=config.seq_len)
    elif network == 'soyak':
        using_spatial = True
        model = Soyak(patch_size=config.patch_size, seq_len=config.seq_len)
    elif network == 'patch_size':
        using_spatial = True
        model = PatchSizeTest(seq_len=config.seq_len, patch_size=config.patch_size)
    elif network == 'balsiger':
        using_spatial = True
        model = Balsiger(seq_len=config.seq_len, patch_size=config.patch_size)
    elif network == 'rca_unet':
        using_spatial = True
        model = RCAUNet(seq_len=config.seq_len, patch_size=config.patch_size,
                        temporal_features=config.num_temporal_features)
    elif network == 'r2plus1d':
        using_spatial = True
        model = R2Plus1D(patch_size=config.patch_size, seq_len=config.seq_len, factorise=config.factorise,
                         non_local_level=config.non_local_level)
    elif network == 'r2plus1d_cbam':
        using_spatial = True
        using_attention = config.cbam_attention or config.rcab_attention
        model = R2Plus1DCbam(patch_size=config.patch_size, seq_len=config.seq_len, factorise=config.factorise,
                             cbam=config.cbam_attention, rcab=config.rcab_attention)
    elif network == 'r2plus1d_non_local':
        using_spatial = True
        model = R2Plus1DNonLocal(patch_size=config.patch_size, seq_len=config.seq_len, factorise=config.factorise)
    elif network == 'r2plus1d_temporal_non_local':
        using_spatial = True
        model = R2Plus1DTemporalNonLocal(patch_size=config.patch_size, seq_len=config.seq_len, factorise=config.factorise)
    else:
        import sys  # Should not be able to reach here as we provide a choice.
        print("Invalid network. Exiting...")
        sys.exit(1)
    return model, using_spatial, using_attention


def main(args, config, logger):
    repo = git.Repo(search_parent_directories=True)
    if not config.debug and repo.is_dirty(submodules=False):
        logger.info("Git head is not clean")

    file_limit = config.limit_number_files if args.file_limit < 0 else args.file_limit
    # If true, return type from model.forward() is ((batch_size, labels), attention)
    model, using_spatial, using_attention = get_network(args.network, config)

    export_dir = get_exports_dir(model, args)
    file_handler = logging.FileHandler(f"{export_dir}/logs.log")
    logger.addHandler(file_handler)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {args.network} model.")
    logger.info(f"Debug mode is {'enabled' if args.debug else 'disabled'}.")
    logger.info(f"There are {args.num_workers} sub-process workers loading training data.")
    logger.info(f"Using device: {device}.")
    logger.info(f"Using {config.seq_len} dimensional fingerprints.")
    logger.info(f"Model: {model}")
    logger.info(f"Args: {args}")
    logger.info(f"Number model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

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
                                export_dir,
                                seq_len=config.seq_len,
                                num_training_dataloader_workers=args.num_workers,
                                num_testing_dataloader_workers=args.num_workers // 2,
                                plot_every=args.plot_every,
                                using_attention=using_attention,
                                using_spatial=using_spatial,
                                valid_chunks=args.chunks,
                                debug=args.debug,
                                limit_number_files=file_limit,
                                limit_iterations=config.limit_iterations,
                                device=device)

    if args.resume_dir is not None:
        # Monkey patch the old model to resume from. Writing a class method turned out to be too
        # tedious as lots of edge cases! Note this does break some things so use with caution.
        path = args.resume_dir + '/Models/' + args.resume_model
        checkpoint = torch.load(path)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimiser = optim.Adam(model.parameters())
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        trainer.lr_scheduler = trainer.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        trainer.epoch = checkpoint['epoch']
        os.remove(trainer.export_dir)
        trainer.export_dir = path

        trainer.starting_epoch = checkpoint['epoch']
        trainer.total_epochs = 5000  # Large number for number of epochs. dont stop

    trainer.loop(args.skip_valid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    network_choices = ['cohen', 'oksuz_rnn', 'hoppe', 'song', 'rnn_attention', 'balsiger',
                       'rca_unet', 'patch_size', 'soyak',
                       'r2plus1d', 'r2plus1d_cbam', 'r2plus1d_non_local', 'r2plus1d_temporal_non_local']
    parser.add_argument('-network', '-n', dest='network', choices=network_choices, type=str.lower, required=True)  # Which network to use.
    parser.add_argument('-debug', '-d', action='store_true', default=False)  # Debug mode. Ignore git warning, get debug logging and custom file limit for debugging.
    parser.add_argument('-workers', '-num_workers', '-w', dest='num_workers', default=0, type=int)  # Number of data loader workers.
    parser.add_argument('-skip_valid', '-no_valid', '-nv', dest='skip_valid', action='store_true', default=False)  # Don't validate.
    parser.add_argument('-plot', '-plot_every', '-plotevery', dest='plot_every', default=1, type=int)  # Plot brain scans every n epochs.
    parser.add_argument('-noplot', '-no_plot', dest='no_plot', action='store_true', default=False)  # Doesn't plot brain scans.
    parser.add_argument('-notes', '-note', dest='notes', type=str)  # Add a note. Used in file dir name.
    parser.add_argument('-cpu', action='store_true', default=False)  # Force to use cpu.
    parser.add_argument('-chunks', default=10, type=int)  # How many chunks to do a validation scan in.
    parser.add_argument('-file_limit', default=-1, type=int)  # Limit number of scans to open at one time.
    parser.add_argument('-resume_dir', default=None, type=str, required=False)
    parser.add_argument('-resume_model', type=int, required=False, default=None)  # Experimental resuming model training.

    args = parser.parse_args()
    args.plot_every = 0 if args.no_plot else args.plot_every

    config = Configuration(args.network, "config.ini", args.debug)

    with setup_logging(config.debug) as logger:
        main(args, config, logger)
