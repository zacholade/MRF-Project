import numpy as np
from abc import abstractmethod


class BaseTransform:
    def __init__(self):
        ...

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class OnlyT1T2(BaseTransform):
    """
    Removes proton density field from the labels.
    Shape (230 x 230 x 3) -> (230 x 230 x 2)
    """
    def __call__(self, sample):
        data, labels, pos = sample
        return data, labels[:2], pos


class ScaleLabels(BaseTransform):
    """
    Scales the T1 and T2 labels by a factor.
    Factor of 1000 converts to milliseconds.
    Ensure proton density is excluded prior to applying this transform.
    """
    def __init__(self, scaling_factor):
        super().__init__()
        self.scaling_factor = scaling_factor

    def __call__(self, sample):
        data, labels, pos = sample
        labels[0] *= self.scaling_factor  # T1
        labels[1] *= self.scaling_factor  # T2
        return data, labels, pos


class NoiseTransform(BaseTransform):
    """
    Adds Gaussian distributed noise to the fingerprint.
    """
    def __init__(self, mean, sd):
        super().__init__()
        self.mean = mean
        self.sd = sd

    def __call__(self, sample):
        data, labels, pos = sample
        noise = np.random.normal(self.mean, self.sd, data.shape)
        data = data + noise
        return data, labels, pos


class ApplyPD(BaseTransform):
    """
    Undoes the normalisation before scaling the fingerprints by proton density.
    Finally it re-normalises them such that the sum of squares == 1.
    """
    def __call__(self, sample):
        data, label, pos = sample
        t1, t2, pd, dn = label
        # old_sum_squares = np.sum(data**2, axis=1)  # To asset if sum of squares == 1

        data *= dn[:, np.newaxis]  # Un-normalise data by the dict norm value per pixel
        data *= pd[:, np.newaxis]  # Apply PD to the un-normalised data
        new_dict_norm = np.sqrt(np.sum(np.abs(np.square(data)), axis=1))  # Calculate new normalisation value per fp.
        data /= new_dict_norm[:, np.newaxis]  # Apply normalisation value to data
        # new_sum_squares = np.sum(data**2, axis=1)  # To assert sum of squares == 1
        return data, label, pos
