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
        data, labels, pos, file_name = sample
        return data, labels[:2], pos, file_name


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
        data, labels, pos, file_name = sample
        labels[0] *= self.scaling_factor  # T1
        labels[1] *= self.scaling_factor  # T2
        return data, labels, pos, file_name


class NoiseTransform(BaseTransform):
    """
    Adds Gaussian distributed noise to the fingerprint.
    """
    def __init__(self, mean, sd):
        super().__init__()
        self.mean = mean
        self.sd = sd

    def __call__(self, sample):
        data, labels, pos, file_name = sample
        noise = np.random.normal(self.mean, self.sd, data.shape)
        data = data + noise
        return data, labels, pos, file_name


class Unnormalise(BaseTransform):
    def __call__(self, sample):
        data, label, pos, file_name = sample
        t1, t2, pd, dn = label
        # old_sum_squares = np.sum(data**2, axis=1)  # To asset if sum of squares == 1
        data *= dn[:, np.newaxis]  # Un-normalise data by the dict norm value per pixel
        return data, label, pos, file_name


class Normalise(BaseTransform):
    def __call__(self, sample):
        data, label, pos, file_name = sample
        new_dict_norm = np.sqrt(np.sum(np.abs(np.square(data)), axis=1))  # Calculate new normalisation value per fp.
        new_dict_norm[new_dict_norm == 0] = 1  # If a value is 0, replace it with 1 or else divide by 0 in next line.
        data /= new_dict_norm[:, np.newaxis]  # Apply normalisation value to data
        label[3] = new_dict_norm  # Update dn value in memory incase we need to do some re-normalisation stuff?
        # new_sum_squares = np.sum(data**2, axis=1)  # To assert sum of squares == 1
        # print(new_sum_squares)
        return data, label, pos, file_name


class ApplyPD(BaseTransform):
    """
    Undoes the normalisation before scaling the fingerprints by proton density.
    Finally it re-normalises them such that the sum of squares == 1.
    """
    def __call__(self, sample):
        data, label, pos, file_name = sample
        t1, t2, pd, dn = label
        data *= pd[:, np.newaxis]  # Apply PD to the un-normalised data
        return data, label, pos, file_name

class Abs(BaseTransform):
    def __call__(self, sample):
        data, label, pos, file_name = sample
        return np.abs(data), label, pos, file_name

class SNRTransform(BaseTransform):
    """
    Adds noise corresponding to a certain SNR ratio.
    """
    def __init__(self, snr, powers: dict = None):
        """
        snr: desired SNR in dB
        """
        super().__init__()
        self.snr_dB = snr  # Desired noise SNR
        self.powers = powers

    def __call__(self, sample):
        """
        x: np.ndarray: Signal to add noise to. Can be of any shape.
        """
        data, label, pos, file_name = sample

        # Calculate linear SNR
        snr = 10.0 ** (self.snr_dB / 10.0)

        # Calculate power of signal using variance
        if self.powers is None:
            signal_power = data.flatten().var()
        else:
            signal_power = self.powers[file_name]

        # Calculate noise power to achieve desired SNR
        noise_power = signal_power / snr
        # Generate noise with calculated power that achieves SNR
        noise = np.random.randn(*data.shape) * np.sqrt(noise_power)

        # Add noise to signal
        data = data + noise

        return data, label, pos, file_name
