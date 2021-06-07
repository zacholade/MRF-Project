import numpy as np
import matplotlib.pyplot as plt
from itertools import islice

import typing

from mrf_map import MRFMap

fp_folder = "Data/MRF_maps/ExactFingerprintMaps/Train/"
parameter_folder = "Data/MRF_maps/ParameterMaps/Train/"
fp_file = "subj1_fisp_slc1_1.npz"
parameter_file = "subj1_fisp_slc1_1.npy"

with open(fp_folder + fp_file, "rb") as f:
    fp = np.load(f)
    fp = fp['arr_0']

with open(parameter_folder + parameter_file, "rb") as f:
    scan = np.load(f)




def batched_data(generator: typing.Iterable, batch_size: int = 1):
    return list(islice(generator, batch_size))


mrf_map = MRFMap.from_scan_and_fp(scan, fp)
data_generator = iter(mrf_map)




# x = np.asarray([i for i in range(1000)])
# for i in range(50, 150):
#     plt.scatter(x, fp[150][i], s=1)
#     plt.show()
