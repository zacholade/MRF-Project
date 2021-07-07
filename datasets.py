from typing import Callable

import numpy as np
import torch.utils.data


class PixelwiseDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, file_lens, transform: Callable = None):
        super().__init__()
        self.transform = transform

        self.labels = labels
        self.data = data
        self._file_lens = file_lens
        self._cum_file_lens = np.cumsum(self._file_lens)
        self._num_total_pixels = np.sum(self._file_lens)

    def __len__(self):
        return self._num_total_pixels

    def __getitem__(self, index):
        index = np.asarray(index)
        file_index = np.argmin((index[:, np.newaxis] // self._cum_file_lens), axis=1)
        pixel_index = index % (self._cum_file_lens[file_index - 1])
        data = self.data[file_index, pixel_index]
        label = self.labels[file_index, pixel_index]
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        data = self.data[index][:self._file_lens[index]]  # second index just removes the padding applied.
        labels = self.labels[index][:self._file_lens[index]]
        t1, t2, pd, pos = labels.transpose()
        labels = np.stack([t1, t2, pd], axis=0).transpose()

        if self.transform:
            data, labels, pos = self.transform((data, labels, pos))

        return data, labels, pos

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        data = torch.FloatTensor(batch[0][0])
        labels = torch.FloatTensor(batch[0][1])
        pos = torch.FloatTensor(batch[0][2])
        return [data, labels, pos]
