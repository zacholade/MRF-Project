import argparse

import numpy as np
import scipy.io
import torch
import torch.utils.data
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler

from config_parser import Configuration
from models.dm import DM
from mrf_trainer import get_network


class DictDataset(torch.utils.data.Dataset):
    """
    Given a collection of scans, randomly fingerprints and labels on a pixelwise
    basis. Can be combined with a random sampler to get random fingerprints.
    """
    def __init__(self, seq_len):
        super().__init__()
        dic = scipy.io.loadmat('Data/Dict/dict.mat')
        lut = scipy.io.loadmat('Data/Dict/lut.mat')
        dn = scipy.io.loadmat('Data/Dict/dn.mat')
        self.dic = torch.FloatTensor(np.array(dic.get('dict')).transpose()).cuda()
        self.lut = torch.FloatTensor(np.array(lut.get('lut')).astype(np.int32)[:, :2]).cuda()
        dn = torch.FloatTensor(np.array(dn.get('dict_norm')).transpose()).cuda()

        print(self.dic.shape)
        print(self.lut.shape)

    def __len__(self):
        return self.lut.shape[0]

    def __getitem__(self, index):
        return self.dic[index], self.lut[index]

    @staticmethod
    def collate_fn(batch):
        data = torch.FloatTensor(batch[0][0])
        labels = torch.FloatTensor(batch[0][1])
        return [data, labels]


def main(args, config):
    dataset = DictDataset(seq_len=300)
    data_loader = DataLoader(dataset,
                             num_workers=0,
                             batch_sampler=BatchSampler(SequentialSampler(dataset), batch_size=100, drop_last=False))


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

    predicted = None
    for current_iteration, (data, lut) in enumerate(iter(data_loader)):
        batch_size = lut.shape[0]
        data = torch.repeat_interleave(data, 9, dim=1).view(batch_size, 300, 3, 3)
        pred = model.forward(data)

        if isinstance(predicted, tuple):  # Attention present
            predicted = predicted[0]

        if predicted is None:
            predicted = pred.detach().cpu().numpy()
        else:
            predicted = np.concatenate((predicted, pred.detach().cpu().numpy()), axis=0)

    lut = dataset.lut.cpu().numpy()

    fig, ax = plt.subplots(2, 2, figsize=(24, 14))

    ax[0][0].grid(True)
    ax[1][1].grid(True)
    ax[0][1].grid(True)
    ax[1][0].grid(True)

    ax[0][0].scatter(lut[:, 0], predicted[:, 0], s=2, label="Estimation")
    ax[1][0].scatter(lut[:, 1], predicted[:, 1], s=2, label="Estimation")

    t1_groundtruth_errors = []
    t2_groundtruth_errors = []
    t1_len = 0
    t2_len = 0
    first_t1 = None
    first_t2 = None
    for t1, t2 in lut:
        if first_t1 is None:
            first_t1 = t1

        if first_t2 is None:
            first_t2 = t2

        if t1 == first_t1:
            t1_len += 1

        if t2 == first_t2:
            t2_len += 1

        if t1 not in t1_groundtruth_errors:
            t1_groundtruth_errors.append(t1)

        if t2 not in t2_groundtruth_errors:
            t2_groundtruth_errors.append(t2)

    t1_predicted_points = [[] for _ in range(len(t1_groundtruth_errors))]
    t2_predicted_points = [[] for _ in range(len(t2_groundtruth_errors))]
    for i, (t1_pred, t2_pred) in enumerate(predicted):
        t1_act, t2_act = lut[i]
        t1_index = t1_groundtruth_errors.index(t1_act)
        t2_index = t2_groundtruth_errors.index(t2_act)

        t1_predicted_points[t1_index].append(t1_pred)
        t2_predicted_points[t2_index].append(t2_pred)

    t1_groundtruth_errors = np.array(t1_groundtruth_errors)
    t2_groundtruth_errors = np.array(t2_groundtruth_errors)

    # Plot red groundtruth lines
    ax[0][0].plot(t1_groundtruth_errors, t1_groundtruth_errors, color='red', label="Ground-Truth")
    ax[1][0].plot(t2_groundtruth_errors, t2_groundtruth_errors, color='red', label="Ground-Truth")
    ax[0][0].legend(loc="upper left", prop={'size': 20})
    ax[1][0].legend(loc="upper left", prop={'size': 20})

    t1_groundtruth_errors = t1_groundtruth_errors.repeat(t1_len)
    t2_groundtruth_errors = t2_groundtruth_errors.repeat(t2_len)
    t1_predicted_points = np.array(t1_predicted_points)
    t2_predicted_points = np.array(t2_predicted_points)
    t1_groundtruth_errors = t1_groundtruth_errors.reshape(t1_predicted_points.shape)
    t2_groundtruth_errors = t2_groundtruth_errors.reshape(t2_predicted_points.shape)

    t1_loss = ((t1_groundtruth_errors - t1_predicted_points) / t1_groundtruth_errors) * 100
    t2_loss = ((t2_groundtruth_errors - t2_predicted_points) / t2_groundtruth_errors) * 100

    ax[0][1].scatter(t1_groundtruth_errors, -t1_loss,  s=2)
    ax[1][1].scatter(t2_groundtruth_errors, -t2_loss, s=2)

    ax[0][0].set_ylabel("Estimated T1 (ms)", family='Arial', fontsize=20)
    ax[0][0].set_xlabel("Groundtruth T1 (ms)", family='Arial', fontsize=20)
    ax[1][0].set_ylabel("Estimated T2 (ms)", family='Arial', fontsize=20)
    ax[1][0].set_xlabel("Groundtruth T2 (ms)", family='Arial', fontsize=20)

    ax[0][1].set_xlabel("Groundtruth T1 (ms)", family='Arial', fontsize=20)
    ax[1][1].set_xlabel("Groundtruth T2 (ms)", family='Arial', fontsize=20)
    ax[0][1].set_ylabel("Relative Error of T1 (ms)", family='Arial', fontsize=20)
    ax[1][1].set_ylabel("Relative Error of T2 (ms)", family='Arial', fontsize=20)

    ax[0][0].set_xlim((200, 3000))
    ax[0][0].set_ylim((200, 3500))

    ax[1][0].set_xlim((50, 500))
    ax[1][0].set_ylim((50, 700))

    ax[0][1].set_xlim((200, 3000))
    ax[1][1].set_xlim((50, 500))

    ax[1][1].tick_params(axis='both', which='major', labelsize=15)
    ax[0][0].tick_params(axis='both', which='major', labelsize=15)
    ax[1][0].tick_params(axis='both', which='major', labelsize=15)
    ax[0][1].tick_params(axis='both', which='major', labelsize=15)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    network_choices = ['cohen', 'oksuz_rnn', 'hoppe', 'song', 'rnn_attention',
                       'patch_size', 'balsiger', 'st', 'dm', 'patch_size', 'rca_unet', 'soyak',
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
