import os

import numpy as np
from matplotlib import pyplot as plt


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

    data_files = np.zeros((len(data_file_names), max_size, 1000))
    label_files = np.zeros((len(label_file_names), max_size, 4))
    file_lens = []
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
        file_lens.append(data_file.shape[0])
        data_files[i] = padded_data_file
        label_files[i] = padded_label_file
        # We want to apply padding to all the fingerprints so that they can be stacked in a big numpy array.
        # Why? Because when we pass this data to our PixelwiseDataset/Scanwise etc, we want to use a batch sampler
        # so that we can sample x amount of indices at once, instead of calling __getitem__ x amount of times (slow).
        # If we cant stack them then we have to store them in a python list which wont let us index it.
        # As such, we then also need to return the file lens (without padding len) so we know to ignore the padded 0s.

    return data_files, label_files, np.asarray(file_lens)


def plot(predicted, labels, pos, save_dir: str = None):
    """
    :param predicted: The predicted t1 and t2 labels.
    :param labels: The ground-truth t1 and t2 labels.
    :param pos: The index position matrix for each t1 and t2 value.
    :param save_dir: Optional argument. Saves the plots to that directory if not None.
    """
    print("Plotting")
    predicted_t1, predicted_t2 = predicted.cpu().detach().numpy().transpose()
    actual_t1, actual_t2 = labels.cpu().numpy().transpose()

    x = (pos // 230).cpu().numpy().astype(int)
    y = (pos % 230).cpu().numpy().astype(int)

    predicted_t1_map, predicted_t2_map = np.zeros((230, 230)), np.zeros((230, 230))
    actual_t1_map, actual_t2_map = np.zeros((230, 230)), np.zeros((230, 230))
    predicted_t1_map[x, y] = predicted_t1
    predicted_t2_map[x, y] = predicted_t2
    actual_t1_map[x, y] = actual_t1
    actual_t2_map[x, y] = actual_t2

    plt.matshow(predicted_t1_map)
    plt.title("Predicted T1")
    plt.clim(0, 3000)
    plt.colorbar(shrink=0.8, label='milliseconds')

    plt.matshow(actual_t1_map)
    plt.title("Actual T1")
    plt.clim(0, 3000)
    plt.colorbar(shrink=0.8, label='milliseconds')

    plt.matshow(np.abs(actual_t1_map - predicted_t1_map))
    plt.title("abs(predicted - actual) T1")
    plt.clim(0, 3000)
    plt.colorbar(shrink=0.8, label='milliseconds')

    plt.matshow(predicted_t2_map)
    plt.title("Predicted T2")
    plt.clim(0, 300)
    plt.colorbar(shrink=0.8, label='milliseconds')

    plt.matshow(actual_t2_map)
    plt.title("Actual T2")
    plt.clim(0, 300)
    plt.colorbar(shrink=0.8, label='milliseconds')

    plt.matshow(np.abs(actual_t2_map - predicted_t2_map))
    plt.title("abs(predicted - actual) T2")
    plt.clim(0, 300)
    plt.colorbar(shrink=0.8, label='milliseconds')

    plt.show()
