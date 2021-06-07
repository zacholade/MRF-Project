from typing import Iterator

import numpy as np
import matplotlib.pyplot as plt
import itertools

import typing

from mrf_map import MRFMap
import torch.utils.data


fp_folder = "Data/MRF_maps/ExactFingerprintMaps/Train/"
parameter_folder = "Data/MRF_maps/ParameterMaps/Train/"
fp_file = "subj1_fisp_slc1_1.npz"
parameter_file = "subj1_fisp_slc1_1.npy"

with open(fp_folder + fp_file, "rb") as f:
    fp = np.load(f)
    fp = fp['arr_0']

with open(parameter_folder + parameter_file, "rb") as f:
    scan = np.load(f)


class TemporalScan(torch.utils.data.IterableDataset):
    def __init__(self, dataset: MRFMap, batch_size: int = 1):
        super().__init__()
        self.dataset = dataset
        self._iter = iter(dataset)
        self.batch_size = batch_size

    def __iter__(self) -> Iterator:
        """
        Iterates over all data on a per-pixel basis
        """
        for t1_p, t2_p, pd_p, fp_p in iter(self.dataset):
            if t1_p != 0 and t2_p != 0:
                yield t1_p, t2_p, pd_p, fp_p

    def __getitem__(self, index):
        ...



mrf_map = MRFMap.from_scan_and_fp(scan, fp)
batch_size = 10
data_generator = iter(mrf_map)
next_10_elements = list(itertools.islice(data_generator, batch_size))




# x = np.asarray([i for i in range(1000)])
# for i in range(50, 150):
#     plt.scatter(x, fp[150][i], s=1)
#     plt.show()
