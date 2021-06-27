import numpy as np


class ExcludeProtonDensity:
    """
    Removes proton density field from the labels.
    Shape (230 x 230 x 3) -> (230 x 230 x 2)
    """
    def __init__(self):
        ...

    def __call__(self, sample):
        data, labels = sample
        if data.shape == (230, 230, 1000):
            t1, t2, pd = np.transpose(labels, axes=(2, 0, 1))
            labels = np.asarray([t1, t2])
            labels = np.transpose(labels, axes=(1, 2, 0))
            return data, labels
        return data, labels[:-1]


class ScaleLabels:
    """
    Scales the T1 and T2 labels by a factor.
    Factor of 1000 converts to milliseconds.
    Ensure proton density is excluded prior to applying this transform.
    """
    def __init__(self, scaling_factor):
        self.scaling_factor = scaling_factor

    def __call__(self, sample):
        # labels = np.transpose(labels, axes=(2, 0, 1))
        # t1, t2 = labels[0] * self.scaling_factor, labels[1] * self.scaling_factor
        # labels = np.asarray([t1, t2])
        # labels = np.transpose(labels, axes=(1, 2, 0))
        data, labels = sample
        return data, labels * self.scaling_factor


class NoiseTransform:
    """
    Adds Gaussian distributed noise to the fingerprint.
    """
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd

    def __call__(self, sample):
        data, labels = sample
        noise = np.random.normal(self.mean, self.sd, data.shape)
        data = data + noise
        return data, labels
