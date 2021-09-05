import scipy.io
import numpy as np
import os


for filename in os.listdir("..Data/Uncompressed/Test/Data"):
    if filename.startswith("."):
        continue

    label = np.load("..Data/Uncompressed/Test/Labels/" + filename)
    data = np.load("..Data/Uncompressed/Test/Data/" + filename)
    dn = label[:, :, 4]
    data *= dn[:, :, np.newaxis]

    label_mat = {"T1_phantom": label[:, :, 0], "T2_phantom": label[:, :, 1],
                 "density": label[:, :, 2], "norm_values": label[:, :, 4]}

    data_mat = {"fingerprint": data}

    filename = filename.replace('.npy', '.mat')
    scipy.io.savemat(f"..CoverBLIP/CoverBLIP toolbox/data/labels/{filename}", label_mat)
    scipy.io.savemat(f"..CoverBLIP/CoverBLIP toolbox/data/data/{filename}", data_mat)

