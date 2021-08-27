import gc
import logging
import os
from multiprocessing import Process

import git
import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


logger = logging.getLogger("mrf")


def get_all_data_files(compressed: bool, test: bool = False, complex_path: str = None):
    compressed = "Compressed" if compressed else "Uncompressed"
    fingerprint_path = f"Data/{compressed}/Data/"
    label_path = f"Data/{compressed}/Labels/"
    if test:
        if complex_path:  # reconstruct scans with COMPLEX noise and undersampled!
            # fingerprint_path = fingerprint_path.
            fingerprint_path = f"Data/{compressed}/Test/ComplexNoise/{complex_path}/"
            label_path = f"Data/{compressed}/Test/Labels/"
        else:
            fingerprint_path = fingerprint_path.replace("/Data", "/Test/Data")
            label_path = label_path.replace("/Labels", "/Test/Labels")

    fingerprint_files = sorted([file for file in os.listdir(fingerprint_path) if not file.startswith(".")])
    label_files = sorted([file for file in os.listdir(label_path) if not file.startswith(".")])

    for file in fingerprint_files:
        if file not in label_files:
            print(file)

    for file in label_files:
        if file not in fingerprint_files:
            raise RuntimeError(f"File: {file} not in other folder!!!")

    for file in fingerprint_files:
        if file not in label_files:
            raise RuntimeError(f"File: {file} not in other folder!!!")

    if len(fingerprint_files) != len(label_files):
        raise RuntimeError("Differing data inside Data/Label folders!")

    # Shuffle two lists the same way.
    to_shuffle = list(zip(fingerprint_files, label_files))
    np.random.shuffle(to_shuffle)
    fingerprint_files, label_files = zip(*to_shuffle)

    fingerprint_files = list(map(lambda file: f"{fingerprint_path}{file}", fingerprint_files))
    label_files = list(map(lambda file: f"{label_path}{file}", label_files))

    return fingerprint_files, label_files


def uncompressed_gen_data(data_names, label_names, max_size: int, seq_len: int):
    file_lens = []
    data_files = np.zeros((len(data_names), 230, 230, seq_len))
    label_files = np.zeros((len(label_names), 230, 230, 5))

    poses = np.zeros((len(label_names), max_size, 1), dtype=int)
    for i, (data_file_name, label_file_name) in enumerate(zip(data_names, label_names)):
        logger.info(f"Loading file {i + 1} / {len(label_names)}.")
        data_file = np.load(data_file_name)
        data_file = data_file[:, :, :seq_len]  # Limit fingerprint length if specified.
        label_file = np.load(label_file_name)
        file_lens.append(np.count_nonzero(label_file[:, :, 2] != 0))

        m = np.ma.masked_equal(label_file[:, :, 2], 0)
        pos_masked = np.ma.masked_array(label_file[:, :, 3], m.mask)
        pos_compressed = np.ma.compressed(pos_masked)
        padded_pos = np.zeros((max_size, 1))
        padded_pos[:pos_compressed.shape[0], 0] = pos_compressed
        poses[i] = padded_pos

        data_file = np.expand_dims(data_file, axis=0)
        label_file = np.expand_dims(label_file, axis=0)
        data_files[i] = data_file
        label_files[i] = label_file

    # Removes the path and the file extension from the file names. Left with just the literal file name.
    file_names = list(map(lambda x: x.split('/')[-1], map(lambda x: x.split('.')[0], data_names)))
    return data_files, label_files, np.asarray(file_lens), file_names, poses


def compressed_gen_data(data_names, label_names, max_size: int, seq_len: int):
    data_files = np.zeros((len(data_names), max_size, seq_len))
    label_files = np.zeros((len(label_names), max_size, 5))
    file_lens = []
    for i, (data_file_name, label_file_name) in enumerate(zip(data_names, label_names)):
        logger.info(f"Loading file {i + 1} / {len(label_names)}.")
        data_file = np.load(data_file_name)
        data_file = data_file[:, :seq_len]  # Limit fingerprint length if specified.
        label_file = np.load(label_file_name)
        data_shape = (max_size, seq_len)
        label_shape = (max_size, label_file.shape[1])
        padded_data_file = np.zeros(data_shape)
        padded_label_file = np.zeros(label_shape)
        padded_data_file[:data_file.shape[0], :seq_len] = data_file
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


def load_eval_files(seq_len: int = 1000, compressed: bool = True, complex_path: str = None):
    data_file_names, label_file_names = get_all_data_files(compressed, test=True, complex_path=complex_path)

    # Find the max shape so we can apply padding in the following for loop instead of a separate for loop.
    files = [np.load(label_file_name, mmap_mode='r').shape[0] for label_file_name in
             list(map(lambda x: x.replace("Uncompressed", "Compressed"), label_file_names))]
    max_size = max(files)

    if compressed:
        return compressed_gen_data(data_file_names, label_file_names, max_size, seq_len)
    else:
        return uncompressed_gen_data(data_file_names, label_file_names, max_size, seq_len)


def load_all_data_files(seq_len: int = 1000, file_limit: int = -1, compressed: bool = True, debug: bool = False):
    data_file_names, label_file_names = get_all_data_files(compressed)

    if file_limit > 0:
        # If we want to limit number of files open (only for memory saving purposes (testing)).
        data_file_names = data_file_names[:file_limit]
        label_file_names = label_file_names[:file_limit]

    # Find the max shape so we can apply padding in the following for loop instead of a separate for loop.
    files = [np.load(label_file_name, mmap_mode='r').shape[0] for label_file_name in
             list(map(lambda x: x.replace("Uncompressed", "Compressed"), label_file_names))]
    max_size = max(files)
    num_files = len(files)

    shuffle_indices = np.arange(num_files)
    np.random.shuffle(shuffle_indices)
    if debug:
        num_train = int(num_files / 24 * 21)  # With 120 files splits train/test into 105 and 15 respectfully
        num_test = num_files - num_train
    else:
        num_test = 10
        num_train = 35

    if num_test == 0 and num_train > 1:
        num_test += 1
        num_train -= 1

    train_indices, test_indices = shuffle_indices[-num_train:], shuffle_indices[:num_test]
    train_data_file_names = [data_file_names[i] for i in train_indices]
    train_label_file_names = [label_file_names[i] for i in train_indices]
    test_data_file_names = [data_file_names[i] for i in test_indices]
    test_label_file_names = [label_file_names[i] for i in test_indices]

    if compressed:
        return compressed_gen_data(train_data_file_names, train_label_file_names, max_size, seq_len),\
               compressed_gen_data(test_data_file_names, test_label_file_names, max_size, seq_len)
    else:
        return uncompressed_gen_data(train_data_file_names, train_label_file_names, max_size, seq_len),\
               uncompressed_gen_data(test_data_file_names, test_label_file_names, max_size, seq_len)


def get_exports_dir(model, args):
    if not os.path.exists("Exports"):
        os.mkdir("Exports")

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    from datetime import datetime
    date = datetime.today().strftime('%Y-%m-%d_%H-%M')

    path = f"{model.__class__.__name__}_{date}"
    if args.notes is not None:
        path = f"{path}_NOTES-{args.notes}"
    path = f"{path}_GIT-{sha}"
    path = f"DEBUG-{path}" if args.debug else path
    path = f"Exports/{path}"

    # This block of code makes sure the folder saving to is new and not been saved to before.
    if os.path.exists(path):
        num = 1
        while os.path.exists(f"{path}_{num}"):
            num += 1
        path = f"{path}_{num}"

    os.mkdir(path)
    return path


def plot_maps(predicted, labels, pos, epoch: int, save_dir: str, subj_name: str):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    predicted_t1, predicted_t2 = predicted.transpose()
    actual_t1, actual_t2 = labels.transpose()

    x = pos // 230
    y = pos % 230

    predicted_t1_map, predicted_t2_map = np.zeros((230, 230)), np.zeros((230, 230))
    actual_t1_map, actual_t2_map = np.zeros((230, 230)), np.zeros((230, 230))
    predicted_t1_map[x, y] = predicted_t1
    predicted_t2_map[x, y] = predicted_t2
    actual_t1_map[x, y] = actual_t1
    actual_t2_map[x, y] = actual_t2

    fig, ax = plt.subplots(2, 4, figsize=(12, 7))
    fig.subplots_adjust(wspace=0.3)

    cmap = None
    im = ax[0][0].matshow(actual_t1_map, vmin=0, vmax=3000, cmap=cmap)
    ax[0][0].title.set_text("True T1")
    ax[0][0].set_xticks([]), ax[0][0].set_yticks([])
    # https://stackoverflow.com/questions/23876588/matplotlib-colorbar-in-each-subplot
    divider = make_axes_locatable(ax[0][0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, shrink=0.8)

    im = ax[0][1].matshow(predicted_t1_map, vmin=0, vmax=3000, cmap=cmap)
    ax[0][1].title.set_text("Predicted T1")
    ax[0][1].set_xticks([]), ax[0][1].set_yticks([])
    divider = make_axes_locatable(ax[0][1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, shrink=0.8)

    im = ax[0][2].matshow(np.abs(actual_t1_map - predicted_t1_map), vmin=0, vmax=3000, cmap=cmap)
    ax[0][2].title.set_text("abs(predicted - true) T1")
    ax[0][2].set_xticks([]), ax[0][2].set_yticks([])
    divider = make_axes_locatable(ax[0][2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, shrink=0.8)

    im = ax[0][3].matshow(np.abs(actual_t1_map - predicted_t1_map), vmin=0, vmax=200, cmap=cmap)
    ax[0][3].title.set_text("abs(predicted - true) T1")
    ax[0][3].set_xticks([]), ax[0][3].set_yticks([])
    divider = make_axes_locatable(ax[0][3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, shrink=0.8)

    im = ax[1][0].matshow(actual_t2_map, vmin=0, vmax=300, cmap=cmap)
    ax[1][0].title.set_text("True T2")
    ax[1][0].set_xticks([]), ax[1][0].set_yticks([])
    divider = make_axes_locatable(ax[1][0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, shrink=0.8)

    im = ax[1][1].matshow(predicted_t2_map, vmin=0, vmax=300, cmap=cmap)
    ax[1][1].title.set_text("Predicted T2")
    ax[1][1].set_xticks([]), ax[1][1].set_yticks([])
    divider = make_axes_locatable(ax[1][1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, shrink=0.8)

    im = ax[1][2].matshow(np.abs(actual_t2_map - predicted_t2_map), vmin=0, vmax=300, cmap=cmap)
    ax[1][2].title.set_text("abs(predicted - actual) T2")
    ax[1][2].set_xticks([]), ax[1][2].set_yticks([])
    divider = make_axes_locatable(ax[1][2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, shrink=0.8)

    im = ax[1][3].matshow(np.abs(actual_t2_map - predicted_t2_map), vmin=0, vmax=20, cmap=cmap)
    ax[1][3].title.set_text("abs(predicted - actual) T2")
    ax[1][3].set_xticks([]), ax[1][3].set_yticks([])
    divider = make_axes_locatable(ax[1][3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, shrink=0.8)

    plt.savefig(f"{save_dir}/{subj_name}_epoch-{epoch}.svg")

    # Clear the current axes.
    plt.cla()
    # Clear the current figure.
    plt.clf()
    # Closes all the figure windows.
    plt.close('all')
    plt.close(fig)
    gc.collect()


def plot_fp(fingerprint, epoch: int = 0, save_dir=None):

    def _plot_fp(fp_):
        x = np.arange(len(fp_))
        plt.scatter(x, fp_, s=1)

    if isinstance(fingerprint, list):
        for fp in fingerprint:
            _plot_fp(fp)
    else:
        _plot_fp(fingerprint)

    if save_dir is not None:
        plt.savefig(f"{save_dir}/_epoch-{epoch}_fingerprint_attention")
    else:
        plt.show()

    # Clear the current axes.
    plt.cla()
    # Clear the current figure.
    plt.clf()
    # Closes all the figure windows.
    gc.collect()


def plot(func, *args, **kwargs):
    """
    Matplotlib has a memory leak which causes memory to grow in the training loop...
    Do plotting in subprocess so that when subprocess terminates, memory is forcibly released.

    :param predicted: The predicted t1 and t2 labels.
    :param labels: The ground-truth t1 and t2 labels.
    :param pos: The index position matrix for each t1 and t2 value.
    :param save_dir: Optional argument. Saves the plots to that directory if not None.
    """
    p = Process(target=func,
                args=args,
                kwargs=kwargs)
    p.start()
    p.join()


def get_inner_patch(x, patch_diameter: int, use_torch: bool = False):
    """
    Input tensor of shape (batch_size, orig_patch, orig_patch, features)
    returns tensor of shape (batch_size, patch_diameter, patch_diameter, features).
    essentially shrinks patch to new size around center! Key is that it takes place in center.
    """
    orig_patch_size = x.shape[1]
    central_index = orig_patch_size // 2
    batch_size = x.shape[0]
    if use_torch:
        spatial_xs = torch.tile(
            torch.arange(central_index - patch_diameter // 2, central_index + 1 + patch_diameter // 2, 1),
            (patch_diameter,))
        spatial_ys = torch.repeat_interleave(
            torch.arange(central_index - patch_diameter // 2, central_index + 1 + patch_diameter // 2, 1),
            patch_diameter)
        x = x[:, spatial_xs, spatial_ys, :].view(batch_size, patch_diameter, patch_diameter, -1)
    else:  # Np array
        spatial_xs = np.tile(
            np.arange(central_index - patch_diameter // 2, central_index + 1 + patch_diameter // 2, 1),
            patch_diameter)
        spatial_ys = np.repeat(
            np.arange(central_index - patch_diameter // 2, central_index + 1 + patch_diameter // 2, 1),
            patch_diameter)
        x = x[:, spatial_xs, spatial_ys, :].reshape(batch_size, patch_diameter, patch_diameter, -1)

    return x