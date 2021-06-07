import numpy as np
from torch.utils.data import IterableDataset, DataLoader


class PixelwiseDataset(IterableDataset):
    def __init__(self,
                 data,
                 labels,
                 batch_size: int = 0,
                 shuffle: bool = False,
                 *args,
                 **kwargs):
        super().__init__()
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

        self.indices = None
        self._shuffle = shuffle

    @property
    def num_columns(self):
        return self.labels.shape[0]

    @property
    def num_rows(self):
        return self.labels.shape[1]

    @property
    def num_pixels(self):
        return self.num_columns * self.num_rows

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value: bool):
        self.indices = np.arange(self.num_pixels) if value else None

    def __len__(self):
        return (self.num_columns * self.num_rows) // self.batch_size

    def __getitem__(self, index):
        index = self.indices[index] if self.shuffle else index
        x = index // self.num_columns
        y = index % self.num_columns
        return self.data[x][y], self.labels[x][y]

    def __iter__(self):
        """
        Iterates through t1/t2/pd/fp on a per pixel basis.
        :return: t1_p, t2_p, pd_p, fp_p
        """
        for i in range(self.num_pixels):
            yield self[i]

    @classmethod
    def from_file_name(cls, filename, *args, **kwargs):
        fingerprint_path, fingerprint_extension = "Data/MRF_maps/ExactFingerprintMaps/Train/", ".npz"
        with open(f"{fingerprint_path}{filename}{fingerprint_extension}", "rb") as f:
            data = np.load(f)['arr_0']

        parameter_path, parameter_extension = "Data/MRF_maps/ParameterMaps/Train/", ".npy"
        with open(f"{parameter_path}{filename}{parameter_extension}", "rb") as f:
            labels = np.load(f)

        # Multiply by 1000 to convert to ms
        labels = np.transpose(labels, axes=(2, 0, 1))
        t1, t2, pd = labels[0] * 1000, labels[1] * 1000, labels[2]
        labels = np.asarray([t1, t2, pd])
        labels = np.transpose(labels, axes=(2, 0, 1))
        return cls(data, labels, *args, **kwargs)

