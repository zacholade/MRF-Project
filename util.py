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

    # Find the max shape so we can apply padding in the following for loop instead of a separate for loop.
    max_size = max([np.load(label_file_name, mmap_mode='r').shape[0] for label_file_name in label_file_names])

    data_files, label_files, file_lens = [], [], []
    for i, (data_file_name, label_file_name) in enumerate(zip(data_file_names, label_file_names)):
        print(f"Loading {data_type}ing file {i+1} / {len(label_file_names)}.")
        data_file = np.load(data_file_name)
        label_file = np.load(label_file_name)
        data_shape = (max_size, data_file.shape[1])
        label_shape = (max_size, label_file.shape[1])
        padded_data_file = np.zeros(data_shape)
        padded_label_file = np.zeros(label_shape)
        padded_data_file[:data_file.shape[0], :data_file.shape[1]] = data_file
        padded_label_file[:label_file.shape[0], :label_file.shape[1]] = label_file
        data_files.append(padded_data_file)
        label_files.append(padded_label_file)
        file_lens.append(data_file.shape[0])

    # We want to apply padding to all the fingerprints so that they can be stacked in a big numpy array.
    # Why? Because when we pass this data to our PixelwiseDataset/Scanwise etc, we want to use a batch sampler
    # so that we can sample x amount of indices at once, instead of calling __getitem__ x amount of times (slow).
    # If we cant stack them then we have to store them in a python list which wont let us index it.
    # As such, we then also need to return the file lens (without padding len) so we know to ignore the padded 0s.
    data_files = np.stack(data_files, axis=0)
    label_files = np.stack(label_files, axis=0)

    return data_files, label_files, np.asarray(file_lens)
