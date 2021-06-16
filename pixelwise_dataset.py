import os

import numpy as np
import torch.utils.data


def get_all_data_files(folder: str = "Train", *args, **kwargs):
    fingerprint_path = f"Data/MRF_maps/ExactFingerprintMaps/{folder}/"
    parameter_path = f"Data/MRF_maps/ParameterMaps/{folder}/"
    fingerprint_files = sorted([file for file in os.listdir(fingerprint_path) if not file.startswith(".")])
    parameter_files = sorted([file for file in os.listdir(parameter_path) if not file.startswith(".")])
    if len(fingerprint_files) != len(parameter_files):
        raise RuntimeError("Differing data inside Test/Train folders!")

    fingerprint_files = list(map(lambda file: f"{fingerprint_path}{file}", fingerprint_files))
    parameter_files = list(map(lambda file: f"{parameter_path}{file}", parameter_files))
    return fingerprint_files, parameter_files


class PixelwiseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_type: str = "Train",
                 transform=None):
        super().__init__()
        self._data_file_names, self._label_file_names = get_all_data_files(data_type)
        self.transform = transform

        # print('before load')
        self._label_files = self._data_files = None
        # self._label_files = [np.load(i, mmap_mode="r") for i in self._label_file_names]
        # self._label_files = self.get_label
        # print('mid load')
        # self._data_files = [np.load(i, mmap_mode="r") for i in self._data_file_names]
        # self._data_files = self.get_data
        # print('after load')

    @property
    def num_indexes(self):
        return self.num_files * self.num_rows * self.num_columns

    @property
    def num_files(self):
        return len(self._data_file_names)

    @property
    def num_pixels_per_matrix(self):
        return self.num_columns * self.num_rows

    @property
    def num_columns(self):
        return 230

    @property
    def num_rows(self):
        return 230

    @property
    def num_timepoints(self):
        return 1000

    def get_2d_index(self, index):
        x = index // self.num_columns
        y = index % self.num_columns
        return x, y

    def __len__(self):
        return self.num_indexes

    def __getitem__(self, index):
        # transform indexes into file index
        file_index = index // self.num_pixels_per_matrix
        pixel_index = index % self.num_pixels_per_matrix
        x, y = self.get_2d_index(pixel_index)

        data = self._data_files[file_index][x][y]
        label = self._label_files[file_index][x][y]

        if self.transform:
            data, label = self.transform((data, label))

        return data, label

    def worker_init_fn(self, worker_id):
        self._data_files = [np.load(i, mmap_mode="r") for i in self._data_file_names]
        self._label_files = [np.load(i, mmap_mode="r") for i in self._label_file_names]

    @staticmethod
    def collate_fn(batch):
        # index = np.array(index)
        # file_index = index // 52900
        # pixel_index = index % 52900
        # x = pixel_index // 230
        # y = pixel_index % 230
        #
        # sorting = np.argsort(file_index)
        # unsorting = np.arange(len(sorting))[sorting]
        # file_index = file_index[sorting]
        # pixel_index = pixel_index[sorting]
        # x = x[sorting]
        # y = y[sorting]
        #
        # curr_label_file = curr_data_file = None
        # curr_file_i = None
        # labels = []
        # datas = []
        #
        # x_s = y_s = []
        # for i, file_i in enumerate(file_index):
        #     if curr_file_i != file_i and curr_file_i is not None:
        #
        #
        #
        #         curr_label_file = self._label_file(file_i)
        #         curr_data_file = self._data_file(file_i)
        #
        #
        #     print(x[i])
        #     print(y[i])
        #     label = curr_label_file[np.ix_(x[i], y[i])]
        #     data = curr_data_file[np.ix_(x[i], y[i])]
        #     labels.append(label)
        #     datas.append(data)

        # Unsort the data / labels
        # datas = torch.FloatTensor(datas[unsorting])
        # labels = torch.FloatTensor(labels[unsorting])
        # return [datas, labels]

        data = torch.FloatTensor([item[0] for item in batch])
        labels = torch.FloatTensor([item[1] for item in batch])
        return [data, labels]
