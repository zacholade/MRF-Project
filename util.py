import os

import numpy as np
from matplotlib import pyplot as plt


def get_all_data_files():
    fingerprint_path = f"Data/Data/"
    label_path = f"Data/Labels/"
    fingerprint_files = sorted([file for file in os.listdir(fingerprint_path) if not file.startswith(".")])
    label_files = sorted([file for file in os.listdir(label_path) if not file.startswith(".")])
    if len(fingerprint_files) != len(label_files):
        raise RuntimeError("Differing data inside Data/Label folders!")

    # Shuffle two lists the same way.
    to_shuffle = list(zip(fingerprint_files, label_files))
    np.random.shuffle(to_shuffle)
    fingerprint_files, label_files = zip(*to_shuffle)

    fingerprint_files = list(map(lambda file: f"{fingerprint_path}{file}", fingerprint_files))
    label_files = list(map(lambda file: f"{label_path}{file}", label_files))

    return fingerprint_files, label_files


def load_all_data_files(file_limit: int = -1):
    data_file_names, label_file_names = get_all_data_files()

    if file_limit > 0:
        # If we want to limit number of files open (only for memory saving purposes (testing)).
        data_file_names = data_file_names[:file_limit]
        label_file_names = label_file_names[:file_limit]

    # Find the max shape so we can apply padding in the following for loop instead of a separate for loop.
    files = [np.load(label_file_name, mmap_mode='r').shape[0] for label_file_name in label_file_names]
    max_size = max(files)
    num_files = len(files)

    shuffle_indices = np.arange(num_files)
    np.random.shuffle(shuffle_indices)
    num_train = int(num_files / 24 * 21)  # With 120 files splits train/test into 105 and 15 respectfully
    num_test = num_files - num_train

    if num_test == 0 and num_train > 1:
        num_test += 1
        num_train -= 1

    train_indices, test_indices = shuffle_indices[-num_train:], shuffle_indices[:num_test]
    train_data_file_names = [data_file_names[i] for i in train_indices]
    train_label_file_names = [label_file_names[i] for i in train_indices]
    test_data_file_names = [data_file_names[i] for i in test_indices]
    test_label_file_names = [label_file_names[i] for i in test_indices]

    def gen_data(data_names, label_names):
        data_files = np.zeros((len(data_names), max_size, 1000))
        label_files = np.zeros((len(label_names), max_size, 4))
        file_lens = []
        for i, (data_file_name, label_file_name) in enumerate(zip(data_names, label_names)):
            print(f"Loading file {i+1} / {len(label_names)}.")
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

        # Removes the path and the file extension from the file names. Left with just the literal file name.
        file_names = list(map(lambda x: x.split('/')[-1], map(lambda x: x.split('.')[0], data_names)))
        return data_files, label_files, np.asarray(file_lens), file_names

    return gen_data(train_data_file_names, train_label_file_names), gen_data(test_data_file_names, test_label_file_names)


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
