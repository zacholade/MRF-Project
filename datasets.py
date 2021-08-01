from typing import Callable

import numpy as np
import torch.utils.data


class PixelwiseDataset(torch.utils.data.Dataset):
    """
    Given a collection of scans, randomly fingerprints and labels on a pixelwise
    basis. Can be combined with a random sampler to get random fingerprints.
    """
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
    """
    Takes in a collection of compressed scans, one index returns one entire scan in a compressed form.
    """
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
    """
    When provided with an index, this dataset will return a spatial collection of neighbouring
    fingerprints about that location. One patch of fingerprints per fingerprint.
    Eg with a patchsize of 5 (that is 5x5). Each index will return 25 fingerprints/labels.
    """
    def __init__(self, patch_size: int, pos, *args, **kwargs):
        super().__init__(*args, **kwargs)
        padding_width = patch_size // 2  # padding width on each side. In-case patch goes out of frame of the image.
        self.padding_width = padding_width
        self.labels = np.pad(self.labels,
                             pad_width=((0, 0), (padding_width, padding_width), (padding_width, padding_width), (0, 0)))
        self.data = np.pad(self.data,
                           pad_width=((0, 0), (padding_width, padding_width), (padding_width, padding_width), (0, 0)))
        self.patch_size = patch_size ** 2
        self.pos = pos

    def __len__(self):
        return self._num_total_pixels

    def __getitem__(self, index):
        index = np.array(index)
        batch_size = index.shape[0]
        file_index = np.argmin((index[:, np.newaxis] // self._cum_file_lens), axis=1)
        datapoint_index = index % (self._cum_file_lens[file_index - 1])
        pixel_index = self.pos[file_index, datapoint_index].squeeze(1)
        x = pixel_index // 230 + self.padding_width  # apply padding width offset!
        y = pixel_index % 230 + self.padding_width
        patch_diameter = int(np.sqrt(self.patch_size))
        patch_radius = patch_diameter // 2
        spatial_xs = np.repeat(x, self.patch_size) + np.tile(np.tile(np.arange(
            0 - patch_radius, 1 + patch_radius, 1), patch_diameter), batch_size)
        spatial_ys = np.repeat(y, self.patch_size) + np.tile(np.repeat(np.arange(
            0 - patch_radius, 1 + patch_radius, 1), patch_diameter), batch_size)
        data = self.data[np.repeat(file_index, self.patch_size),
                         spatial_xs,
                         spatial_ys, :]
        label = self.labels[np.repeat(file_index, self.patch_size),
                            spatial_xs,
                            spatial_ys, :]
        pos = label[:, 3]
        label = np.delete(label, 3, axis=1).transpose()

        if self.transform:
            data, label, pos = self.transform((data, label, pos))

        label = label.transpose()
        label = label[int(self.patch_size // 2)::self.patch_size, :]  # Label is the central pixel
        pos = pos[int(self.patch_size // 2)::self.patch_size]  # Pos is the central pixel
        data = data.reshape(batch_size, patch_diameter, patch_diameter, -1).transpose((0, 3, 1, 2))
        return data, label, pos


class ScanPatchDataset(PatchwiseDataset):
    """
    Extends Patchwise Dataset but now indexes are alligned with scans. Because patch sampling
    increases the number of pixels by a factor of patchsize squared, you can provide a chunking
    parameter. A chunk of a whole scan will be returned. Num fingerprints = Scan size // chunk size.
    """
    def __init__(self, chunks: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        pixel_index = self.pos[file_index, datapoint_index].squeeze(1)
        x = pixel_index // 230 + self.padding_width  # apply padding width offset!
        y = pixel_index % 230 + self.padding_width
        patch_diameter = int(np.sqrt(self.patch_size))
        patch_radius = patch_diameter // 2
        spatial_xs = np.repeat(x, self.patch_size) + np.tile(np.tile(np.arange(
            0 - patch_radius, 1 + patch_radius, 1), patch_diameter), batch_size)
        spatial_ys = np.repeat(y, self.patch_size) + np.tile(np.repeat(np.arange(
            0 - patch_radius, 1 + patch_radius, 1), patch_diameter), batch_size)
        data = self.data[np.repeat(file_index, self.patch_size * batch_size),
                         spatial_xs,
                         spatial_ys, :]
        label = self.labels[np.repeat(file_index, self.patch_size * batch_size),
                            spatial_xs,
                            spatial_ys, :]
        pos = label[:, 3]
        label = np.delete(label, 3, axis=1).transpose()

        if self.transform:
            data, label, pos = self.transform((data, label, pos))

        label = label.transpose()
        label = label[int(self.patch_size // 2)::self.patch_size, :]  # Label is the central pixel
        pos = pos[int(self.patch_size // 2)::self.patch_size]  # Pos is the central pixel
        data = data.reshape(batch_size, patch_diameter, patch_diameter, -1).transpose((0, 3, 1, 2))
        return data, label, pos, file_name
