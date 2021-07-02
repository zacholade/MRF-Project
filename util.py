import os

import numpy as np


def get_all_data_files(folder: str = "Train", *args, **kwargs):
    fingerprint_path = f"Data/MRF_maps/Data/{folder}/"
    parameter_path = f"Data/MRF_maps/Labels/{folder}/"
    fingerprint_files = sorted([file for file in os.listdir(fingerprint_path) if not file.startswith(".")])
    parameter_files = sorted([file for file in os.listdir(parameter_path) if not file.startswith(".")])
    if len(fingerprint_files) != len(parameter_files):
        raise RuntimeError("Differing data inside Test/Train folders!")

    fingerprint_files = list(map(lambda file: f"{fingerprint_path}{file}", fingerprint_files))
    parameter_files = list(map(lambda file: f"{parameter_path}{file}", parameter_files))
    return fingerprint_files, parameter_files


def load_all_data_files(data_type: str = "Train", file_limit: int = -1):
    data_file_names, label_file_names = get_all_data_files(data_type)

    if file_limit > 0:
        # If we want to limit number of files open (only for memory saving purposes (testing)).
        data_file_names = data_file_names[:file_limit]
        label_file_names = label_file_names[:file_limit]

    label_files, data_files = [], []
    for i, (data_file_name, label_file_name) in enumerate(zip(data_file_names, label_file_names)):
        print(f"Loading {data_type}ing file {i+1} / {len(label_file_names)}.")
        data_files.append(np.load(data_file_name))
        label_files.append(np.load(label_file_name))

    return data_files, label_files
