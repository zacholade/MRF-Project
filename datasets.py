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
        pos = label[:, 3]
        label = np.delete(label, 3, axis=1).transpose()

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
        label = self.labels[index][:self._file_lens[index]]
        file_name = self._file_names[index]
        pos = label[:, 3]
        label = np.delete(label, 3, axis=1).transpose()

        if self.transform:
            data, label, pos = self.transform((data, label, pos))

        return data, label.transpose(), pos, file_name

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
    _3x3_offsets = np.array([-231, -230, -229, -1, 0, 1, 229, 230, 231])
    _5x5_offsets = np.array([
        -462, -461, -460, -459, -258,
        -232, -231, -230, -229, -228,
        -2, -1, 0, 1, 2,
        228, 229, 230, 231, 232,
        458, 459, 460, 461, 462])

    def __init__(self, patch_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size ** 2
        self.offsets = self._3x3_offsets if patch_size == 3 else self._5x5_offsets
        if patch_size not in [3, 5]:
            raise ValueError("Invalid patch size")

    def __len__(self):
        return self._num_total_pixels

    def _get_surrounding_datapoint_indexes(self, file_index, datapoint_index):
        batch_size = file_index.shape[0]
        # the -2 value gets the second to last value in the 3rd dimension (pos).
        central_pixel_index = self.labels[file_index, datapoint_index, -2]

        # Create two arrays ith the shape (batch_size, patch_size)
        # 9 = 3x3 array from top left to top right and then down for each row.
        # Pixel index is used to store all the datapoints we will use.
        # The central pixel index is just used to apply a mask to any pixel indexes
        # which go out of bounds once applying the _offsets matrix (defined above).
        pixel_index = np.repeat(central_pixel_index[:, np.newaxis], self.patch_size, axis=1)
        central_pixel_index = np.repeat(central_pixel_index[:, np.newaxis], self.patch_size, axis=1)

        # Apply the offsets as per the _offsets matrix defined above.
        pixel_index = pixel_index + self.offsets
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
        batch_size = index.shape[0]
        file_index = np.argmin((index[:, np.newaxis] // self._cum_file_lens), axis=1)
        datapoint_index = index % (self._cum_file_lens[file_index - 1])
        spatial_datapoint_index = self._get_surrounding_datapoint_indexes(file_index, datapoint_index)
        data = self.data[np.repeat(file_index, self.patch_size), spatial_datapoint_index]
        label = self.labels[np.repeat(file_index, self.patch_size), spatial_datapoint_index]
        pos = label[:, 3]
        label = np.delete(label, 3, axis=1).transpose()

        if self.transform:
            data, label, pos = self.transform((data, label, pos))

        label = label.transpose()
        label = label[int(np.floor(self.patch_size // 2))::self.patch_size, :]
        pos = pos[int(np.floor(self.patch_size // 2))::self.patch_size]
        data = data.reshape(batch_size, -1, int(np.sqrt(self.patch_size)), int(np.sqrt(self.patch_size)))
        return data, label, pos


class ScanPatchDataset(PatchwiseDataset):
    def __init__(self, chunks, patch_size, data, labels, *args, **kwargs):
        super().__init__(patch_size, data, labels, *args, **kwargs)
        self.chunks = chunks

    def __len__(self):
        return len(self.data) * self.chunks

    def __getitem__(self, index):
        chunk = index % self.chunks
        file_index = index // self.chunks
        if chunk + 1 == self.chunks:  # Last chunk of that file. Might not be same size.
            _file_len = self._file_lens[file_index]
            batch_size = _file_len - ((self.chunks - 1) * (_file_len // self.chunks))
        else:
            batch_size = self._file_lens[file_index] // self.chunks
        file_name = self._file_names[file_index]
        datapoint_index = np.arange(batch_size) + (chunk * (self._file_lens[file_index] // self.chunks))
        file_index = np.repeat(np.asarray(file_index), batch_size)
        spatial_datapoint_index = self._get_surrounding_datapoint_indexes(file_index, datapoint_index)
        data = self.data[np.repeat(file_index, self.patch_size), spatial_datapoint_index]
        label = self.labels[np.repeat(file_index, self.patch_size), spatial_datapoint_index]
        pos = label[:, 3]
        label = np.delete(label, 3, axis=1).transpose()

        if self.transform:
            data, label, pos = self.transform((data, label, pos))

        label = label.transpose()
        label = label[int(np.floor(self.patch_size // 2))::self.patch_size, :]
        pos = pos[int(np.floor(self.patch_size // 2))::self.patch_size]
        data = data.reshape(batch_size, -1, int(np.sqrt(self.patch_size)), int(np.sqrt(self.patch_size)))
        return data, label, pos, file_name

    def _get_surrounding_datapoint_indexes(self, file_index, datapoint_index):
        # the -2 value gets the second to last value in the 3rd dimension (pos).
        central_pixel_index = self.labels[file_index, datapoint_index, -2]

        # Create two arrays ith the shape (batch_size, patch_size)
        # 9 = 3x3 array from top left to top right and then down for each row.
        # Pixel index is used to store all the datapoints we will use.
        # The central pixel index is just used to apply a mask to any pixel indexes
        # which go out of bounds once applying the _offsets matrix (defined above).
        pixel_index = np.repeat(central_pixel_index[:, np.newaxis], self.patch_size, axis=1)
        central_pixel_index = np.repeat(central_pixel_index[:, np.newaxis], self.patch_size, axis=1)

        # Apply the offsets as per the _offsets matrix defined above.
        pixel_index = pixel_index + self.offsets
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
