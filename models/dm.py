import h5py
import numpy as np
import torch
from torch import nn


class DM(nn.Module):
    """
    Implements dictionary matching.
    """
    def __init__(self, seq_len):
        super().__init__()
        dm_file = h5py.File("Data/dict.mat", 'r')
        self.lut = torch.Tensor(np.array(dm_file.get('lut'))).cuda()
        self.dic = torch.FloatTensor(np.array(dm_file.get('dict'))).cuda()

        # Dict is already normalised. If using fingerprints less than 1000 timesteps,
        # unnormalise, shorten and then renormalise so energy = 1.
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
