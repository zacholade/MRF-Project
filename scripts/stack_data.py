import numpy as np
from util import get_all_data_files, load_all_data_files
import pickle


def refactor():
    train_data_file_names, train_label_file_names = get_all_data_files("Train")
    test_data_file_names, test_label_file_names = get_all_data_files("Test")

    data_file_names = [*train_data_file_names, *test_data_file_names]
    label_file_names = [*train_label_file_names, *test_label_file_names]

    data_file_names = data_file_names[:5]
    label_file_names = label_file_names[:5]

    # Find the max shape so we can apply padding in the following for loop instead of a separate for loop.
    max_size = max([np.load(label_file_name, mmap_mode='r').shape[0] for label_file_name in label_file_names])

    data_files = np.zeros((len(data_file_names), max_size, 1000))
    label_files = np.zeros((len(label_file_names), max_size, 4))
    file_lens = []
    for i, (data_file_name, label_file_name) in enumerate(zip(data_file_names, label_file_names)):
        print(f"Loading file {i+1} / {len(label_file_names)}.")
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
    file_names = list(map(lambda x: x.split('/')[-1], map(lambda x: x.split('.')[0], data_file_names)))

    with open("Data/data.npy", "wb") as f:
        np.save(f, data_files)

    with open("Data/labels.npy", "wb") as f:
        np.save(f, label_files)

    with open("Data/file_lens.npy", "wb") as f:
        np.save(f, np.asarray(file_lens))

    with open("Data/file_names.pickle", "wb") as f:
        pickle.dump(file_names, f)

    return data_files, label_files, np.asarray(file_lens), file_names


refactor()
