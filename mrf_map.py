import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PixelwiseDataset(Dataset):
    def __init__(self,
                 data,
                 labels,
                 *args,
                 **kwargs):
        super().__init__()
        self.data = data
        self.labels = labels

    @property
    def num_columns(self):
        return self.labels.shape[0]

    @property
    def num_rows(self):
        return self.labels.shape[1]

    @property
    def num_pixels(self):
        return self.num_columns * self.num_rows

    def __len__(self):
        return self.num_pixels

    def __getitem__(self, index):
        x = index // self.num_columns
        y = index % self.num_columns
        return self.data[x][y], self.labels[x][y]

    @staticmethod
    def collate_fn(batch):
        data = torch.FloatTensor([item[0] for item in batch])
        labels = torch.FloatTensor([item[1] for item in batch])
        return [data, labels]

    @classmethod
    def from_file_name(cls, filename, *args, **kwargs):
        fingerprint_path, fingerprint_extension = "Data/MRF_maps/ExactFingerprintMaps/Train/", ".npz"
        with open(f"{fingerprint_path}{filename}{fingerprint_extension}", "rb") as f:
            data = np.load(f)['arr_0']

        parameter_path, parameter_extension = "Data/MRF_maps/ParameterMaps/Train/", ".npy"
        with open(f"{parameter_path}{filename}{parameter_extension}", "rb") as f:
            labels = np.load(f)

        # Unpack t1 and t2 so that we can multiply by 1000 to convert to ms.
        # Then repack to original shape.
        labels = np.transpose(labels, axes=(2, 0, 1))
        t1, t2, pd = labels[0] * 1000, labels[1] * 1000, labels[2]
        labels = np.asarray([t1, t2, pd])
        labels = np.transpose(labels, axes=(1, 2, 0))
        return cls(data, labels, *args, **kwargs)


if __name__ == "__main__":
    dataset_loader = DataLoader(PixelwiseDataset.from_file_name("subj1_fisp_slc4_4"),
                                batch_size=1000,
                                shuffle=True,
                                pin_memory=True,
                                collate_fn=PixelwiseDataset.collate_fn)

    i = iter(dataset_loader)
    data, labels = next(i)
    data.cuda()
