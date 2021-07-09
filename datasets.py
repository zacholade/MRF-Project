from typing import Callable

import numpy as np
import torch.utils.data


class PixelwiseDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, file_lens, file_names, set_indices,
                 transform: Callable = None, mmap: bool = False):
        super().__init__()
        self.transform = transform

        self.labels = labels
        self.data = data
        self._file_lens = file_lens[set_indices]
        self._file_names = [file_names[i] for i in set_indices]
        self._cum_file_lens = np.cumsum(self._file_lens)
        self._num_total_pixels = np.sum(self._file_lens)
        self._set_indices = set_indices  # The random indices used to determine train/valid dataset
        self.mmap = mmap

    def worker_init(self, i):
        if self.mmap:

            self.data = np.load("Data/data.npy", mmap_mode="r")
            self.labels = np.load("Data/data.npy", mmap_mode="r")

    def __len__(self):
        return self._num_total_pixels

    def __getitem__(self, index):
        index = np.asarray(index)
        print(self.labels[self._set_indices].shape)
        import time
        time.sleep(10)
        file_index = np.argmin((index[:, np.newaxis] // self._cum_file_lens), axis=1)
        pixel_index = index % (self._cum_file_lens[file_index - 1])
        data = self.data[self._set_indices][file_index, pixel_index]
        label = self.labels[self._set_indices][file_index, pixel_index]
        t1, t2, pd, pos = label.transpose()
        label = np.stack([t1, t2, pd], axis=0).transpose()

        if self.transform:
            data, label, pos = self.transform((data, label, pos))

        return data, label, pos

    @staticmethod
    def collate_fn(batch):
        data = torch.FloatTensor(batch[0][0])
        labels = torch.FloatTensor(batch[0][1])
        pos = torch.FloatTensor(batch[0][2])
        return [data, labels, pos]


class ScanwiseDataset(PixelwiseDataset):
    """1 index = 1 entire scan."""
    def __getitem__(self, index):
        data = self.data[self._set_indices][index][:self._file_lens[index]]  # second index just removes the padding applied.
        labels = self.labels[self._set_indices][index][:self._file_lens[index]]
        file_name = self._file_names[index]
        t1, t2, pd, pos = labels.transpose()
        labels = np.stack([t1, t2, pd], axis=0).transpose()

        if self.transform:
            data, labels, pos = self.transform((data, labels, pos))

        return data, labels, pos, file_name

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        data = torch.FloatTensor(batch[0][0])
        labels = torch.FloatTensor(batch[0][1])
        pos = torch.FloatTensor(batch[0][2])
        file_name = batch[0][3]
        return [data, labels, pos, file_name]
