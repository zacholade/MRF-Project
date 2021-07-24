from typing import Callable

import numpy as np
import torch.utils.data


class PixelwiseDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, file_lens, file_names, transform: Callable = None):
        super().__init__()
        self.transform = transform

        self.labels = labels
        self.data = data
        self._file_lens = file_lens
        self._file_names = file_names
        self._cum_file_lens = np.cumsum(self._file_lens)
        self._num_total_pixels = np.sum(self._file_lens)

    def __len__(self):
        return self._num_total_pixels

    def __getitem__(self, index):
        index = np.asarray(index)
        file_index = np.argmin((index[:, np.newaxis] // self._cum_file_lens), axis=1)
        datapoint_index = index % (self._cum_file_lens[file_index - 1])
        data = self.data[file_index, datapoint_index]
        label = self.labels[file_index, datapoint_index]
        t1, t2, pd, pos, dn = label.transpose()
        label = np.stack([t1, t2, pd, dn], axis=0)

        if self.transform:
            data, label, pos = self.transform((data, label, pos))

        return data, label.transpose(), pos

    @staticmethod
    def collate_fn(batch):
        data = torch.FloatTensor(batch[0][0])
        labels = torch.FloatTensor(batch[0][1])
        pos = torch.FloatTensor(batch[0][2])
        return [data, labels, pos]


class ScanwiseDataset(PixelwiseDataset):
    """1 index = 1 entire scan."""
    def __getitem__(self, index):
        data = self.data[index][:self._file_lens[index]]  # second index just removes the padding applied.
        labels = self.labels[index][:self._file_lens[index]]
        file_name = self._file_names[index]
        t1, t2, pd, pos, dn = labels.transpose()
        labels = np.stack([t1, t2, pd, dn], axis=0)

        if self.transform:
            data, labels, pos = self.transform((data, labels, pos))

        return data, labels.transpose(), pos, file_name

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        data = torch.FloatTensor(batch[0][0])
        labels = torch.FloatTensor(batch[0][1])
        pos = torch.FloatTensor(batch[0][2])
        file_name = batch[0][3]
        return [data, labels, pos, file_name]


class PatchwiseDataset(PixelwiseDataset):
    _offsets = np.array([-231, -230, -229, -1, 0, 1, 229, 230, 231])

    def __len__(self):
        return self._num_total_pixels

    def _get_surrounding_datapoint_indexes(self, file_index, datapoint_index):
        batch_size = file_index.shape[0]
        # the -2 value gets the second to last value in the 3rd dimension (pos).
        central_pixel_index = self.labels[file_index, datapoint_index, -2]

        # Create two arrays ith the shape (batch_size, 9)
        # 9 = 3x3 array from top left to top right and then down for each row.
        # Pixel index is used to store all the datapoints we will use.
        # The central pixel index is just used to apply a mask to any pixel indexes
        # which go out of bounds once applying the _offsets matrix (defined above).
        pixel_index = np.repeat(central_pixel_index[:, np.newaxis], 9, axis=1)
        central_pixel_index = np.repeat(central_pixel_index[:, np.newaxis], 9, axis=1)

        # Apply the offsets as per the _offsets matrix defined above.
        pixel_index = pixel_index + self._offsets
        pos = self.labels[file_index, :, -2]

        # Any pixels out of bounds (oob)?
        oob_mask = np.invert((pixel_index[..., None] == pos[:, None]).any(2))
        # If so, we can just set the pixel index as the central pixel of the spatial area.
        pixel_index[oob_mask] = central_pixel_index[oob_mask]

        # Now we have pixelpoint indexes. This following line converts it to datapoint indexes.
        # np.where gets the index position of all trues. It returns a tuple for each dimension:
        # which in our case is (batch_size, 9, num of datapoints per scan with padding (+-27000 or something).
        # We only want the last dimension so we index [2] on it.
        datapoint_index = np.where(pos[:, None] == pixel_index[..., None])[2]
        return datapoint_index

    def __getitem__(self, index):
        index = np.asarray(index)
        file_index = np.argmin((index[:, np.newaxis] // self._cum_file_lens), axis=1)
        datapoint_index = index % (self._cum_file_lens[file_index - 1])
        datapoint_index = self._get_surrounding_datapoint_indexes(file_index, datapoint_index)
        data = self.data[np.repeat(file_index, 9), datapoint_index]
        label = self.labels[np.repeat(file_index, 9), datapoint_index]
        t1, t2, pd, pos, dn = label.transpose()
        label = np.stack([t1, t2, pd, dn], axis=0)

        if self.transform:
            data, label, pos = self.transform((data, label, pos))

        return data, label.transpose(), pos
