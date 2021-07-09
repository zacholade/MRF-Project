import os
import pickle

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


def load_all_data_files():
    data = np.load("Data/data.npy")
    labels = np.load("Data/labels.npy")
    file_lens = np.load("Data/file_lens.npy")
    with open("Data/file_names.pickle", "rb") as f:
        file_names = pickle.load(f)

    return data, labels, file_lens, file_names


def plot(predicted, labels, pos, epoch: int, save_dir: str):
    """
    :param predicted: The predicted t1 and t2 labels.
    :param labels: The ground-truth t1 and t2 labels.
    :param pos: The index position matrix for each t1 and t2 value.
    :param save_dir: Optional argument. Saves the plots to that directory if not None.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

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
    plt.savefig(f"{save_dir}/epoch-{epoch}_Pred-t1.png")

    plt.matshow(actual_t1_map)
    plt.title("Actual T1")
    plt.clim(0, 3000)
    plt.colorbar(shrink=0.8, label='milliseconds')
    plt.savefig(f"{save_dir}/epoch-{epoch}_t1.png")

    plt.matshow(np.abs(actual_t1_map - predicted_t1_map))
    plt.title("abs(predicted - actual) T1")
    plt.clim(0, 3000)
    plt.colorbar(shrink=0.8, label='milliseconds')
    plt.savefig(f"{save_dir}/epoch-{epoch}_Pred-True-t1.png")

    plt.matshow(predicted_t2_map)
    plt.title("Predicted T2")
    plt.clim(0, 300)
    plt.colorbar(shrink=0.8, label='milliseconds')
    plt.savefig(f"{save_dir}/epoch-{epoch}_Pred-t2.png")

    plt.matshow(actual_t2_map)
    plt.title("Actual T2")
    plt.clim(0, 300)
    plt.colorbar(shrink=0.8, label='milliseconds')
    plt.savefig(f"{save_dir}/epoch-{epoch}_True-t2.png")

    plt.matshow(np.abs(actual_t2_map - predicted_t2_map))
    plt.title("abs(predicted - actual) T2")
    plt.clim(0, 300)
    plt.colorbar(shrink=0.8, label='milliseconds')
    plt.savefig(f"{save_dir}/epoch-{epoch}_Pred-True-t2.png")

    plt.close('all')
