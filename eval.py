"""
Used to evaluate model performance using the test data.
"""

import math

import torch
import os

from torch import nn

from data_logger import DataLogger
import argparse
from config_parser import Configuration
from models.dm import DM
from mrf_trainer import get_network
from datasets import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.pylab as pl
from matplotlib.ticker import MaxNLocator

from transforms import ApplyPD, OnlyT1T2, NoiseTransform, Normalise, Unnormalise
from util import load_eval_files, plot_maps, plot


def plot_1d_nlocal_attention2(attention, data):
    attention = attention.detach().cpu().numpy()
    batch_index = 10
    attention = attention[batch_index]
    data = data[batch_index]
    attention -= (1 / 300)  # Normalise to 0. Would otherwise be about 0.0033 (1/300)
    plt.tight_layout()
    fig, ax = plt.subplots(6, 1, figsize=(12, 7))

    plt.margins(0)
    plt.xlabel("Time Point (or channel)", labelpad=20)
    fig.text(0.05, 0.35, "Normalised Attention Score", rotation='vertical')

    largest_value = 0
    max_value = 0
    for i, attention_line in enumerate(attention[np.array([0, 60, 120, 180, 240, 299])]):
        max_value = max(np.max(np.abs(attention_line)), max_value)
        print(i)
        # ax[i//60].plot(np.zeros(len(attention_line)), linewidth=1, color='black')
        ax[i].plot(attention_line)

    nearest_005 = math.ceil(max_value * 200) / 200  # Round to nearest 0.005
    for axis in range(6):
        ax[axis].spines['top'].set_visible(False)
        ax[axis].spines['right'].set_visible(False)
        ax[axis].set_ylim([-nearest_005, nearest_005])
        ax[axis].set_xlim([0, 300])
        ax[axis].spines['bottom'].set_position('center')
        # ax[axis].set_xticklabels(ax[axis].get_xticks(), rotation=90)


    # Data plot
    fig_data, ax_data = plt.subplots(1, 1, figsize=(12, 7))
    ax_data.set_ylabel("Normalised fingeprint (a.u.)", family='Arial', fontsize=15)
    ax_data.set_xlabel("Excitation number", family='Arial', fontsize=15)
    plt.margins(0)
    plt.grid()  # linewidth=0.1, color='black')
    ax_data.spines['right'].set_linewidth(0.5)
    ax_data.spines['top'].set_linewidth(0.5)
    ax_data.spines['right'].set_color('grey')
    ax_data.spines['top'].set_color('grey')
    nearest_005 = math.ceil(np.max(np.abs(data)) * 20) / 20  # Round to nearest 0.005
    ax_data.set_ylim([0, nearest_005])
    ax_data.set_xlim([0, 300])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax_data.plot(np.abs(data))

    plt.show()


def plot_1d_nlocal_attention(attention, data):
    attention = attention.detach().cpu().numpy()
    batch_index = 10
    attention = attention[batch_index]
    attention -= (1 / 300)  # Normalise to 0. Would otherwise be about 0.0033 (1/300)
    plt.tight_layout()
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    # with open("out_csv.csv", 'a', newline='') as file:
    my_cmap = cm.viridis
    colors = pl.cm.viridis(np.linspace(0, 1, data.shape[1]))
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=300))
    cbar = plt.colorbar(sm)
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.set_ylabel("Excitation number", labelpad=15, family='Arial', fontsize=15)
    ax[1].set_ylabel("Attention Score (a.u.)", family='Arial', fontsize=15)
    ax[1].set_xlabel("Per timestep attention", family='Arial', fontsize=15)
    plt.margins(0)
    plt.grid()  # linewidth=0.1, color='black')
    ax[1].spines['right'].set_linewidth(0.5)
    ax[1].spines['top'].set_linewidth(0.5)
    ax[1].spines['right'].set_color('grey')
    ax[1].spines['top'].set_color('grey')
    ax[1].set_ylim([np.min(attention[0:299:10].flatten()), np.max(attention[0:299:10].flatten())])
    ax[1].set_xlim([0, 300])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    for i, attention_line in enumerate(attention):
        if i % 10 == 0:
            ax[1].plot(attention_line, color=colors[i], linewidth=1)

    im = ax[0].plot(np.abs(data[batch_index, :].detach().cpu().numpy()))
    plt.show()


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


def remove_zero_labels(predicted, labels, pos=None):
    """
    Some models return a full patch prediction (soyak, rca-unet).
    In these cases, some labels will contain air. Remove these from predicted and labels
    so we dont back prop on them as they are later masked
    and so that we dont result in infinity values for MAPE due to label being zero.
    """
    mask = labels[:, 0] != 0
    if pos is not None:
        return predicted[mask], labels[mask], pos[mask]
    return predicted[mask], labels[mask]


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

    loss_func = nn.MSELoss()

    if args.cs is not None:
        complex_path = f"snr-{args.snr}_cs-{args.cs}"
        seq_len = 200
        data_transforms = transforms.Compose(
            [Unnormalise(), ApplyPD(), Normalise(), NoiseTransform(0, 0.01), OnlyT1T2()])
    else:
        complex_path = None
        seq_len = config.seq_len
        data_transforms = transforms.Compose(
            [Unnormalise(), ApplyPD(), Normalise(), NoiseTransform(0, 0.01), OnlyT1T2()])

    if using_spatial:
        _data, _labels, _file_lens, _file_names, _pos = load_eval_files(seq_len=seq_len,
                                                                        compressed=not using_spatial,
                                                                        complex_path=complex_path)
        validation_dataset = ScanPatchwiseDataset(args.chunks, model.patch_size, _pos, patch_return_size, _data,
                                                  _labels, _file_lens, _file_names, transform=data_transforms)
    else:
        _data, _labels, _file_lens, _file_names = load_eval_files(seq_len=seq_len,
                                                                  compressed=not using_spatial,
                                                                  complex_path=complex_path)
        validation_dataset = ScanwiseDataset(args.chunks, _data, _labels, _file_lens,
                                             _file_names, transform=data_transforms)

    data_loader = torch.utils.data.DataLoader(validation_dataset,
                                              batch_size=1,
                                              collate_fn=ScanwiseDataset.collate_fn,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.workers)

    export_dir = f"Exports/Test/{args.path.split('.')[0]}"
    if args.snr is not None:
        export_dir += f"_snr-{args.snr}_cs-{args.cs}"

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
        attention = None
        predicted = model.forward(data)
        if isinstance(predicted, tuple):
            attention = predicted[1]
            predicted = predicted[0]

        if patch_return_size > 1:
            predicted = get_inner_patch(predicted, patch_return_size, use_torch=True).view(-1, 2)
            predicted, labels, pos = remove_zero_labels(predicted, labels, pos)



        if attention is not None:
            pass
            # attention = attention.cpu().detach().numpy()
            # data = data.detach().cpu().numpy()
            # vmin = np.min(attention[10][0:299:10].flatten())
            # vmax = np.max(attention[10][0:299:10].flatten())
            # attention = attention[10, 0:299:1, 0:299:1]
            # print(attention.shape)
            # plt.matshow(attention, vmin=vmin, vmax=vmax, cmap='hot')
            # plt.show()
            # plot_1d_nlocal_attention2(attention, data.detach().cpu().numpy())
            # plot_1d_nlocal_attention(attention, data)

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

    data_logger.on_epoch_end(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    network_choices = ['cohen', 'oksuz_rnn', 'hoppe', 'song', 'rnn_attention',
                       'patch_size', 'balsiger', 'st', 'dm', 'patch_size',
                       'r2plus1d', 'r2plus1d_cbam', 'r2plus1d_non_local', 'r2plus1d_temporal_non_local']
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
