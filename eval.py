import numpy as np
import torch
import os

from torch import nn

from data_logger import DataLogger
from models import *
import argparse
from config_parser import Configuration
from mrf_trainer import get_network
from datasets import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

#  R2Plus1D_epoch-42_optim-Adam_initial-lr-0.001_loss-MSELoss_batch-size-100.pt
from transforms import ApplyPD, OnlyT1T2
from util import load_eval_files, plot_maps, plot


def main(args, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, using_spatial, using_attention = get_network(args.network, config)
    loss_func = nn.MSELoss()
    checkpoint = torch.load(args.path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    data_transforms = transforms.Compose([ApplyPD(), OnlyT1T2()])
    if using_spatial:
        _data, _labels, _file_lens, _file_names, _pos = load_eval_files(seq_len=config.seq_len,
                                                                        compressed=not using_spatial)
        validation_dataset = ScanPatchDataset(args.chunks, model.patch_size, _pos, _data,
                                              _labels, _file_lens, _file_names, transform=data_transforms)
    else:
        _data, _labels, _file_lens, _file_names = load_eval_files(seq_len=config.seq_len,
                                                                  compressed=not using_spatial)
        validation_dataset = ScanwiseDataset(args.chunks, _data, _labels, _file_lens,
                                             _file_names, transform=data_transforms)

    data_loader = torch.utils.data.DataLoader(validation_dataset,
                                              batch_size=1,
                                              collate_fn=ScanwiseDataset.collate_fn,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.workers)
    export_dir = f"Exports/Test/{args.path.split('.')[0]}"
    if not os.path.exists(export_dir):
        os.mkdir(export_dir)

    data_logger = DataLogger(export_dir)

    validate_set = iter(data_loader)
    current_chunk = 0  # The current chunk of one scan currently processing
    chunk_loss = 0
    chunk_predicted = None
    chunk_labels = None
    chunk_pos = None
    for current_iteration, (data, labels, pos, file_name) in enumerate(validate_set):
        current_chunk += 1
        print(f"Scan: {(current_iteration // args.chunks) + 1} / "
              f"{len(data_loader) // args.chunks}. "
              f"Chunk: {((current_chunk - 1) % args.chunks) + 1} / "
              f"{args.chunks}")
        data, labels = data.to(device), labels.to(device)
        predicted = model.forward(data)
        if isinstance(predicted, tuple):
            attention = predicted[1]
            predicted = predicted[0]
        loss = loss_func(predicted, labels)

        data, labels = data.cpu(), labels.cpu()
        chunk_loss += loss.detach().cpu().item()
        chunk_predicted = predicted.detach().cpu() if chunk_predicted is None else \
            torch.cat((chunk_predicted, predicted.detach().cpu()), 0)
        chunk_labels = labels.detach().cpu() if chunk_labels is None else \
            torch.cat((chunk_labels, labels.detach().cpu()), 0)
        chunk_pos = pos.detach().cpu() if chunk_pos is None else \
            torch.cat((chunk_pos, pos.detach().cpu()), 0)
        if current_chunk % args.chunks == 0:
            data_logger.log_error(chunk_predicted,
                                  chunk_labels,
                                  chunk_loss, "test")

            if not os.path.exists(f"{export_dir}/Plots"):
                os.mkdir(f"{export_dir}/Plots")
            # Matplotlib has a memory leak. To alleviate this do plotting in a subprocess and
            # join to it. When process is suspended, memory is forcibly released.
            plot(plot_maps,
                 chunk_predicted.numpy(),
                 chunk_labels.numpy(),
                 chunk_pos.numpy().astype(int),
                 0,
                 f"{export_dir}/Plots/{file_name}",
                 file_name)

            chunk_loss = 0
            chunk_predicted = None
            chunk_labels = None
            chunk_pos = None

    data_logger.on_epoch_end(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    network_choices = ['cohen', 'oksuz_rnn', 'hoppe', 'song', 'rnn_attention', 'balsiger', 'st', 'r2plus1d']
    parser.add_argument('-network', '-n', dest='network', choices=network_choices, type=str.lower, required=True)
    parser.add_argument('-chunks', default=10, type=int)  # How many chunks to do a validation scan in.
    parser.add_argument('-path', required=True)  # Path to the model + filename
    parser.add_argument('-workers', '-num_workers', '-w', dest='workers', default=0, type=int)
    args = parser.parse_args()

    config = Configuration(args.network, "config.ini", debug=False)

    with torch.no_grad():
        main(args, config)
