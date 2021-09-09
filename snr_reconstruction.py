import argparse
import os

import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from config_parser import Configuration
from data_logger import DataLogger
from datasets import *
from models.dm import DM
from mrf_trainer import get_network
from transforms import OnlyT1T2, Normalise, Unnormalise, SNRTransform
from util import load_eval_files, plot_maps, plot, remove_zero_labels


def log_in_vivo_sections(predicted, labels, data_logger):
    white_matter_mask = torch.where((685-33 <= labels[:, 0]) & (labels[:, 0] <= 685+33), True, False)
    predicted_white_matter_masked = predicted[white_matter_mask]
    true_white_matter_masked = labels[white_matter_mask]

    grey_matter_mask = torch.where((1180-104 <= labels[:, 0]) & (labels[:, 0] <= 1180+104), True, False)
    predicted_grey_matter_masked = predicted[grey_matter_mask]
    true_grey_matter_masked = labels[grey_matter_mask]

    cbsf_mask = torch.where((4880-379 <= labels[:, 0]) & (labels[:, 0] <= 4880+251), True, False)
    predicted_cbsf_masked = predicted[cbsf_mask]
    true_cbsf_masked = labels[cbsf_mask]

    # If statements in case there is no tissue in the range for that scan.
    if true_cbsf_masked.size(0) != 0:
        data_logger.log_error(predicted_white_matter_masked, true_white_matter_masked, None, "white")
    if true_grey_matter_masked.size(0) != 0:
        data_logger.log_error(predicted_grey_matter_masked, true_grey_matter_masked, None, "grey")
    if true_cbsf_masked.size(0) != 0:
        data_logger.log_error(predicted_cbsf_masked, true_cbsf_masked, None, "cbsf")


def main(args, config):
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
    loss_func = nn.MSELoss()

    export_dir = f"Exports/Test/{args.path.split('.')[0]}"
    export_dir += f"_snr_recon"

    if not os.path.exists(export_dir):
        os.mkdir(export_dir)

    data_logger = DataLogger(export_dir)

    complex_path = None
    seq_len = config.seq_len
    if using_spatial:
        _data, _labels, _file_lens, _file_names, _pos = load_eval_files(seq_len=seq_len,
                                                                        compressed=not using_spatial,
                                                                        complex_path=complex_path)
    else:
        _data, _labels, _file_lens, _file_names = load_eval_files(seq_len=seq_len,
                                                                  compressed=not using_spatial,
                                                                  complex_path=complex_path)

    # Calculate power of overall mrf data as opposed to per fp.
    map_powers = {}
    for file_name in _file_names:
        uncomp_data = np.load(f"Data/Uncompressed/Test/Data/{file_name}.npy")
        uncomp_label = np.load(f"Data/Uncompressed/Test/Labels/{file_name}.npy")
        dn = uncomp_label[:, :, 4]
        uncomp_data *= dn[:, :, np.newaxis]
        map_power = uncomp_data.flatten().var()
        map_powers[file_name] = map_power


    monte_carlo_iterations = 25
    for snr in (2, 4, 8, 16, 32, 64):
        for i in range(monte_carlo_iterations):  # 25 simulations per noise.
            print(f"{snr}:{i}")
            data_transforms = transforms.Compose(
                [Unnormalise(), SNRTransform(snr), Normalise(), OnlyT1T2()])

            if using_spatial:
                validation_dataset = ScanPatchwiseDataset(args.chunks, model.patch_size, _pos, patch_return_size, _data,
                                                          _labels, _file_lens, _file_names, transform=data_transforms)
            else:
                validation_dataset = ScanwiseDataset(args.chunks, _data, _labels, _file_lens,
                                                     _file_names, transform=data_transforms)

            print(f"SNR: {snr}, iteration: {i} / {monte_carlo_iterations}")
            data_loader = torch.utils.data.DataLoader(validation_dataset,
                                                      batch_size=1,
                                                      collate_fn=ScanwiseDataset.collate_fn,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=args.workers)
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
                      f"{args.chunks}. SNR: {snr}")
                data, labels = data.to(device), labels.to(device)
                predicted = model.forward(data)
                if isinstance(predicted, tuple):
                    attention = predicted[1]
                    predicted = predicted[0]

                if patch_return_size > 1:
                    predicted = get_inner_patch(predicted, patch_return_size, use_torch=True).view(-1, 2)
                    predicted, labels, pos = remove_zero_labels(predicted, labels, pos)

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

                    plot(plot_maps,
                         chunk_predicted.numpy(),
                         chunk_labels.numpy(),
                         chunk_pos.numpy().astype(int),
                         0,
                         f"{export_dir}/Plots/{file_name}",
                         f"{file_name}_snr-{snr}")

                    chunk_loss = 0
                    chunk_predicted = None
                    chunk_labels = None
                    chunk_pos = None

            data_logger.on_epoch_end(snr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    network_choices = ['cohen', 'oksuz_rnn', 'hoppe', 'song', 'rnn_attention',
                       'patch_size', 'balsiger', 'st', 'dm', 'patch_size', 'rca_unet', 'soyak',
                       'r2plus1d']
    parser.add_argument('-network', '-n', dest='network', choices=network_choices, type=str.lower, required=True)
    parser.add_argument('-chunks', default=10, type=int)  # How many chunks to do a validation scan in.
    parser.add_argument('-path', required=True)  # Path to the model + filename
    parser.add_argument('-full_model', default=False, action='store_true')
    parser.add_argument('-workers', '-num_workers', '-w', dest='workers', default=0, type=int)
    args = parser.parse_args()

    config = Configuration(args.network, "config.ini", debug=False)

    with torch.no_grad():
        main(args, config)
