"""
Used to evaluate model performance using the test data.
"""

import argparse
import os
import time

import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from config_parser import Configuration
from data_logger import DataLogger
from datasets import *
from models.dm import DM
from mrf_trainer import get_network
from transforms import ApplyPD, OnlyT1T2, Normalise, Unnormalise, NoiseTransform
from util import load_eval_files, plot_maps, plot, log_in_vivo_sections, remove_zero_labels


def main(args, config):
    patch_return_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.network.lower() == "dm":
        model, using_spatial, using_attention = DM(config.seq_len), False, False
    else:
        model, using_spatial, using_attention, patch_return_size = get_network(args.network, config)
        if args.full_model:
            model = torch.load(f"FinalModels/{args.path}")
        else:
            checkpoint = torch.load(f"FinalModels/{args.path}")
            model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    # plot_weights(model, 0, single_channel=False)

    loss_func = nn.MSELoss()

    if args.cs is not None:
        undersampled_path = f"cs-{args.cs}"
        seq_len = 200
        data_transforms = transforms.Compose(
            [Normalise(), NoiseTransform(0, 0.01), OnlyT1T2()])
    else:
        undersampled_path = None
        seq_len = config.seq_len
        data_transforms = transforms.Compose(
            [Unnormalise(), ApplyPD(), Normalise(), NoiseTransform(0, 0.01), OnlyT1T2()])

    if using_spatial:
        _data, _labels, _file_lens, _file_names, _pos = load_eval_files(seq_len=seq_len,
                                                                        compressed=not using_spatial,
                                                                        complex_path=undersampled_path)
        validation_dataset = ScanPatchwiseDataset(args.chunks, model.patch_size, _pos, patch_return_size, _data,
                                                  _labels, _file_lens, _file_names, transform=data_transforms)
    else:
        _data, _labels, _file_lens, _file_names = load_eval_files(seq_len=seq_len,
                                                                  compressed=not using_spatial,
                                                                  complex_path=undersampled_path)
        validation_dataset = ScanwiseDataset(args.chunks, _data, _labels, _file_lens,
                                             _file_names, transform=data_transforms)

    data_loader = torch.utils.data.DataLoader(validation_dataset,
                                              batch_size=1,
                                              collate_fn=ScanwiseDataset.collate_fn,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.workers)

    export_dir = f"Exports/Test/{args.path.split('.')[0]}"
    if args.cs is not None:
        export_dir += f"cs-{args.cs}"

    if not os.path.exists(export_dir):
        os.mkdir(export_dir)

    data_logger = DataLogger(export_dir)

    validate_set = iter(data_loader)
    current_chunk = 0  # The current chunk of one scan currently processing
    chunk_loss = 0
    chunk_predicted = None
    chunk_labels = None
    chunk_pos = None

    total_time = 0
    total_iterations = 0
    for current_iteration, (data, labels, pos, file_name) in enumerate(validate_set):
        total_iterations = current_iteration
        print(current_iteration)
        current_chunk += 1
        print(f"Scan: {(current_iteration // args.chunks) + 1} / "
              f"{len(data_loader) // args.chunks}. "
              f"Chunk: {((current_chunk - 1) % args.chunks) + 1} / "
              f"{args.chunks}")
        data, labels = data.to(device), labels.to(device)
        attention = None

        start_time = time.time()
        predicted = model.forward(data)
        total_time += time.time() - start_time

        if isinstance(predicted, tuple):
            attention = predicted[1]
            predicted = predicted[0]

        if patch_return_size > 1:
            predicted = get_inner_patch(predicted, patch_return_size, use_torch=True).view(-1, 2)
            predicted, labels, pos = remove_zero_labels(predicted, labels, pos)

        if attention is not None:
            # plot_1d_nlocal_attention2(attention, data.detach().cpu().numpy())
            # plot_1d_nlocal_attention(attention, data)
            # plot_cbam_attention(attention.detach().cpu().numpy(), data.detach().cpu().numpy())
            ...

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

            log_in_vivo_sections(chunk_predicted, chunk_labels, data_logger)

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

    print("--- Average scan time: %s seconds ---" % (total_time / (total_iterations / args.chunks)))
    data_logger.on_epoch_end(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    network_choices = ['cohen', 'oksuz_rnn', 'hoppe', 'song', 'rnn_attention',
                       'patch_size', 'balsiger', 'st', 'dm', 'patch_size', 'rca_unet', 'soyak',
                       'r2plus1d', 'r1d']
    parser.add_argument('-network', '-n', dest='network', choices=network_choices, type=str.lower, required=True)
    parser.add_argument('-chunks', default=10, type=int)  # How many chunks to do a validation scan in.
    parser.add_argument('-path', required=True)  # Path to the model + filename
    parser.add_argument('-full_model', default=False, action='store_true')
    parser.add_argument('-workers', '-num_workers', '-w', dest='workers', default=0, type=int)
    parser.add_argument('-snr', default=None, type=int)
    parser.add_argument('-cs', default=None, type=int)
    args = parser.parse_args()

    config = Configuration(args.network, "config.ini", debug=False)

    with torch.no_grad():
        main(args, config)
