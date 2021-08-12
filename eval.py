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
from torch.utils.data import DataLoader


from transforms import ApplyPD, OnlyT1T2, NoiseTransform, Normalise, Unnormalise
from util import load_eval_files, plot_maps, plot
import h5py


class DM(nn.Module):
    """
    Implements dictionary matching.
    """
    def __init__(self, seq_len):
        super().__init__()
        dm_file = h5py.File("Data/dict.mat", 'r')
        self.lut = torch.Tensor(np.array(dm_file.get('lut'))).cuda()
        self.dic = torch.FloatTensor(np.array(dm_file.get('dict'))).cuda()
        if seq_len != 1000:
            dn = torch.Tensor(np.array(dm_file.get('dict_norm'))).cuda()
            self.dic *= dn
            self.dic = self.dic[:, :seq_len]
            new_dict_norm = torch.sqrt(torch.sum(torch.abs(torch.square(self.dic)), dim=1)).unsqueeze(1)
            self.dic /= new_dict_norm

    def forward(self, x):
        out = torch.zeros(x.shape[0], 3, device=x.device)
        for i, fingerprint in enumerate(x):
            fingerprint = fingerprint.unsqueeze(1)
            dot = torch.mm(self.dic, fingerprint)
            out[i] = self.lut[:, torch.argmax(dot)]
        return out[:, :2]


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
        model, using_spatial, using_attention = get_network(args.network, config)
        checkpoint = torch.load(args.path)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    loss_func = nn.MSELoss()

    if args.snr is not None:
        complex_path = f"snr-{args.snr}_cs-{args.cs}"
        seq_len = 200
        data_transforms = transforms.Compose(
            [Unnormalise(), ApplyPD(), Normalise(), OnlyT1T2()])
    else:
        complex_path = None
        seq_len = config.seq_len
        data_transforms = transforms.Compose(
            [Unnormalise(), ApplyPD(), Normalise(), NoiseTransform(0, 0.01), OnlyT1T2()])

    if using_spatial:
        _data, _labels, _file_lens, _file_names, _pos = load_eval_files(seq_len=seq_len,
                                                                        compressed=not using_spatial,
                                                                        complex_path=complex_path)
        validation_dataset = ScanPatchwiseDataset(args.chunks, model.patch_size, _pos, _data,
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
                       'patch_size', 'balsiger', 'st', 'r2plus1d', 'dm']
    parser.add_argument('-network', '-n', dest='network', choices=network_choices, type=str.lower, required=True)
    parser.add_argument('-chunks', default=10, type=int)  # How many chunks to do a validation scan in.
    parser.add_argument('-path', required=True)  # Path to the model + filename
    parser.add_argument('-workers', '-num_workers', '-w', dest='workers', default=0, type=int)
    parser.add_argument('-snr', default=None, type=int)
    parser.add_argument('-cs', default=None, type=int)
    args = parser.parse_args()

    config = Configuration(args.network, "config.ini", debug=False)

    with torch.no_grad():
        main(args, config)
