from typing import Callable

import numpy as np
import torch.utils.data

from util import get_inner_patch


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
        self._file_names = np.array(file_names, dtype=object)

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
            data, label, pos, _ = self.transform((data, label, pos, None))

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
        datapoint_index = np.arange(batch_size) + (chunk * (self._file_lens[file_index] // self.chunks))

        data = self.data[np.repeat(file_index, batch_size), datapoint_index]
        label = self.labels[np.repeat(file_index, batch_size), datapoint_index]
        file_name = self._file_names[file_index]
        pos = label[:, 3]
        label = np.delete(label, 3, axis=1).transpose()

        if self.transform:
            data, label, pos, _ = self.transform((data, label, pos, file_name))

        return data, label.transpose(), pos, file_name

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
    def __init__(self, patch_size: int, pos, patch_return_size: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size ** 2
        self.pos = pos
        self.patch_return_size = patch_return_size

        padding_width_top_left = padding_width_bottom_right = patch_size // 2  # padding width on each side. In-case patch goes out of frame of the image.
        if patch_return_size > 1:
            padding_width_bottom_right += (230 % patch_size)

        self.padding_width = padding_width_top_left
        self.labels = np.pad(self.labels,
                             pad_width=((0, 0), (padding_width_top_left, padding_width_bottom_right),
                                        (padding_width_top_left, padding_width_bottom_right), (0, 0)))
        self.data = np.pad(self.data,
                           pad_width=((0, 0), (padding_width_top_left, padding_width_bottom_right),
                                      (padding_width_top_left, padding_width_bottom_right), (0, 0)))

        if patch_return_size > 1:
            self._calculate_new_len()

    @property
    def _patch_gap(self):
        """
        distance between patches. 1 for training so we get all possible patch positions.
        For eval this is the patch return diameter. only takes effect if patches return
        more than just the prediction of the central patch pixel.
        """
        return 1

    def _calculate_new_len(self):
        """
        Helper function to transform data if the model returns a prediction for an entire spatial patch
        as opposed to JUST the central pixel. Eg U-net which returns a large patch prediction
        as opposed to just 1x1 prediction given a input patch.
        """
        patch_diameter = self.patch_return_size
        patch_radius = patch_diameter // 2
        filewise_central_pixel_indices = []
        for label_file in self.labels:
            pd = label_file[:, :, 2]
            masked_map = pd != 0
            central_pixel_indices = []
            for x_pos in range(0, 229 + (230 % patch_diameter), self._patch_gap):
                for y_pos in range(0, 229 + (230 % patch_diameter), self._patch_gap):
                    spatial_xs = np.tile(x_pos + np.arange(0 - patch_radius, 1 + patch_radius, 1), patch_diameter) + self.padding_width
                    spatial_ys = np.repeat(y_pos + np.arange(0 - patch_radius, 1 + patch_radius, 1), patch_diameter) + self.padding_width
                    patch = masked_map[spatial_xs, spatial_ys]
                    if np.any(patch):  # One pixel in the patch is not air.
                        central_pixel_indices.append(x_pos * 230 + (y_pos))
            filewise_central_pixel_indices.append(central_pixel_indices)

        file_lens = []
        for central_pixel_indices in filewise_central_pixel_indices:
            file_lens.append(len(central_pixel_indices))

        self._file_lens = np.array(file_lens, dtype=int)
        self._cum_file_lens = np.cumsum(self._file_lens)
        self._num_total_pixels = np.sum(self._file_lens)

        # Work out max size for padding purposes
        max_size = 0
        for central_pixel_indices in filewise_central_pixel_indices:
            if len(central_pixel_indices) > max_size:
                max_size = len(central_pixel_indices)

        # Apply padding
        for central_pixel_indices in filewise_central_pixel_indices:
            while len(central_pixel_indices) < max_size:
                central_pixel_indices.append(0)

        self._patch_offset_indices = np.array(filewise_central_pixel_indices, dtype=int)

    def __len__(self):
        return self._num_total_pixels

    def __getitem__(self, index):
        index = np.array(index)
        batch_size = index.shape[0]
        file_index = np.argmin((index[:, np.newaxis] // self._cum_file_lens), axis=1)
        datapoint_index = index % (self._cum_file_lens[file_index - 1])

        if self.patch_return_size > 1:
            pixel_index = self._patch_offset_indices[file_index, datapoint_index]
        else:
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
            # Apply transformations such as apply PD, normalise and add noise.
            data, label, pos, _ = self.transform((data, label, pos, None))

        label = label.transpose()

        if self.patch_return_size == 1:
            label = label[int(self.patch_size // 2)::self.patch_size, :]  # Label is the central pixel
            pos = pos[int(self.patch_size // 2)::self.patch_size]  # Pos is the central pixel
        else:
            label = label.reshape(batch_size, patch_diameter, patch_diameter, -1)
            pos = pos.reshape(batch_size, patch_diameter, patch_diameter, -1)
            label = get_inner_patch(label, self.patch_return_size)
            pos = get_inner_patch(pos, self.patch_return_size)
            label = label.reshape(-1, 2)
            pos = pos.reshape(-1)

        data = data.reshape(batch_size, patch_diameter, patch_diameter, -1).transpose((0, 3, 1, 2))
        return data, label, pos


class ScanPatchwiseDataset(PatchwiseDataset):
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

        if self.patch_return_size > 1:
            pixel_index = self._patch_offset_indices[file_index, datapoint_index]
        else:
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
            data, label, pos, _ = self.transform((data, label, pos, file_name))

        label = label.transpose()

        if self.patch_return_size == 1:
            label = label[int(self.patch_size // 2)::self.patch_size, :]  # Label is the central pixel
            pos = pos[int(self.patch_size // 2)::self.patch_size]  # Pos is the central pixel
        else:
            label = label.reshape(batch_size, patch_diameter, patch_diameter, -1)
            pos = pos.reshape(batch_size, patch_diameter, patch_diameter, -1)
            label = get_inner_patch(label, self.patch_return_size)
            pos = get_inner_patch(pos, self.patch_return_size)
            label = label.reshape(-1, 2)
            pos = pos.reshape(-1)
        data = data.reshape(batch_size, patch_diameter, patch_diameter, -1).transpose((0, 3, 1, 2))
        return data, label, pos, file_name

    @property
    def _patch_gap(self):
        return self.patch_return_size
