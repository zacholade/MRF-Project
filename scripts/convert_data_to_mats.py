import scipy.io
import numpy as np
import os


for filename in os.listdir("../Data/Uncompressed/Data"):
    if filename.startswith("."):
        continue

    label = np.load("Data/Uncompressed/Labels/" + filename)
    data = np.load("Data/Uncompressed/Data/" + filename)
    dn = label[:, :, 4]
    data *= dn[:, :, np.newaxis]

    label_mat = {"T1": label[:, :, 0], "T2": label[:, :, 1],
                 "PD": label[:, :, 2], "norm_values": label[:, :, 4]}

    data_mat = {"time_signal": data}

    filename = filename.replace('.npy', '.mat')
    scipy.io.savemat(f"CoverBLIP/CoverBLIP toolbox/data/labels/{filename}", label_mat)
    scipy.io.savemat(f"CoverBLIP/CoverBLIP toolbox/data/data/{filename}", data_mat)

    import sys
    sys.exit()
