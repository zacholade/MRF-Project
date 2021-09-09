"""
This file is used to convert the original 230x230x1000 data
a and 230x230x3 labels into a 1 dimensional format in order
to reduce both memory space and disk size.
"""


import numpy as np
import os


def get_all_data_files(folder: str = "Train", *args, **kwargs):
    fingerprint_path = f"Data/MRF_maps/ExactFingerprintMaps/{folder}/"
    parameter_path = f"Data/MRF_maps/ParameterMaps/{folder}/"
    fingerprint_files = sorted([file for file in os.listdir(fingerprint_path) if not file.startswith(".")])
    parameter_files = sorted([file for file in os.listdir(parameter_path) if not file.startswith(".")])
    if len(fingerprint_files) != len(parameter_files):
        raise RuntimeError("Differing data inside Test/Train folders!")

    fingerprint_files = list(map(lambda file: f"{fingerprint_path}{file}", fingerprint_files))
    parameter_files = list(map(lambda file: f"{parameter_path}{file}", parameter_files))
    return fingerprint_files, parameter_files


data_file_names, label_file_names = get_all_data_files("Test/")

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

indices = np.arange(230 * 230).reshape(230, 230)
for data_file_name, label_file_name in zip(data_file_names, label_file_names):
    data_file = np.load(data_file_name)
    label_file = np.load(label_file_name)
    t1, t2, pd = np.transpose(label_file, axes=(2, 0, 1))
    m = np.ma.masked_equal(pd, 0)
    t1_masked, t2_masked, pd_masked, indices_masked = \
        np.ma.masked_array(t1, m.mask), np.ma.masked_array(t2, m.mask),\
        np.ma.masked_array(pd, m.mask), np.ma.masked_array(indices, m.mask)
    t1_compressed, t2_compressed, pd_compressed, indices_compressed = \
        np.ma.compressed(t1_masked), np.ma.compressed(t2_masked),\
        np.ma.compressed(pd_masked), np.ma.compressed(indices_masked)

    fp_compressed = []
    for index in indices_compressed:
        x = index // 230
        y = index % 230
        fp_compressed.append(data_file[x][y])
    fp_compressed = np.asarray(fp_compressed)

    labels = np.asarray([t1_compressed, t2_compressed, pd_compressed, indices_compressed])
    data = fp_compressed

    to_save_label_file_name = label_file_name.replace("MRF_maps", "MRF_maps/Compressed").replace("ParameterMaps", "Labels")
    to_save_data_file_name = data_file_name.replace("MRF_maps", "MRF_maps/Compressed").replace("ExactFingerprintMaps", "Data")
    with open(to_save_data_file_name, "wb") as f:
        np.save(f, data)

    with open(to_save_label_file_name, "wb") as f:
        np.save(f, labels)


