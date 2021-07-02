from typing import Callable

import numpy as np
import torch.utils.data


class PixelwiseDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform: Callable = None):
        super().__init__()
        self.transform = transform

        self.labels = labels
        self.data = data

        self._file_lens = np.asarray([label_file.shape[1] for label_file in self.labels])
        self._cum_file_lens = np.cumsum(self._file_lens)
        self._num_total_pixels = np.sum(self._file_lens)

    def __len__(self):
        return self._num_total_pixels

    def __getitem__(self, index):
        file_index = np.argwhere((index // self._cum_file_lens) == 0)[0][0]
        pixel_index = index % (self._cum_file_lens[file_index - 1])
        data = self.data[file_index][pixel_index]
        label = (self.labels[file_index][0][pixel_index],
                 self.labels[file_index][1][pixel_index],
                 self.labels[file_index][2][pixel_index])
        pos = self.labels[file_index][3][pixel_index]

        if self.transform:
            data, label, pos = self.transform((data, label, pos))

        return data, label, pos

    @staticmethod
    def collate_fn(batch):
        data = torch.FloatTensor([item[0] for item in batch])
        labels = torch.FloatTensor([item[1] for item in batch])
        pos = torch.FloatTensor([item[2] for item in batch])
        return [data, labels, pos]


class ScanwiseDataset(PixelwiseDataset):
    """1 index = 1 entire scan."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.labels[index]
        t1, t2, pd, pos = labels
        labels = (t1, t2, pd)

        if self.transform:
            data, labels, pos = self.transform((data, labels, pos))

        labels = np.asarray(labels).transpose()
        return data, labels, pos

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        data = torch.FloatTensor(batch[0][0])
        labels = torch.FloatTensor(batch[0][1])
        pos = torch.FloatTensor(batch[0][2])
        return [data, labels, pos]

