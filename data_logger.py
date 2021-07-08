import csv
import os
from collections import defaultdict

import numpy as np
import torch


class DataLogger:
    """
    The idea is to call the log method each iteration of an epoch during both train and validation.
    The value is appended to a list. Then, when on_epoch_end() method is called. It will take the mean of
    all values in the list and save it to a csv file.
    """
    def __init__(self, directory: str):
        self._log = defaultdict(list)
        self._directory = directory

    @property
    def directory(self) -> str:
        return self._directory

    @property
    def filename(self) -> str:
        return f"logs.csv"

    @property
    def qualified_filename(self) -> str:
        return f"{self.directory}/{self.filename}"

    def log_error(self, predicted, labels, loss, data_type: str):
        mean_abs_perc_error = torch.mean(torch.abs(((labels - predicted) / labels))) * 100
        mean_sq_error = torch.mean(((labels - predicted) ** 2))
        root_mean_sq_error = torch.sqrt(mean_sq_error)
        self.log(f"{data_type}_loss", (loss / len(labels)).cpu())
        self.log(f"{data_type}_mape", mean_abs_perc_error.cpu())
        self.log(f"{data_type}_mse", mean_sq_error.cpu())
        self.log(f"{data_type}_rmse", root_mean_sq_error.cpu())

        t1_true, t2_true = torch.transpose(labels, 0, 1)
        t1_pred, t2_pred = torch.transpose(predicted, 0, 1)

        t1_mean_abs_perc_error = torch.mean(torch.abs(((t1_true - t1_pred) / t1_true))) * 100
        t1_mean_sq_error = torch.mean(((t1_true - t1_pred) ** 2))
        t1_root_mean_sq_error = torch.sqrt(t1_mean_sq_error)
        self.log(f"{data_type}_t1_mape", t1_mean_abs_perc_error.cpu())
        self.log(f"{data_type}_t1_mse", t1_mean_sq_error.cpu())
        self.log(f"{data_type}_t1_rmse", t1_root_mean_sq_error.cpu())

        t2_mean_abs_perc_error = torch.mean(torch.abs(((t2_true - t2_pred) / t2_true))) * 100
        t2_mean_sq_error = torch.mean(((t2_true - t2_pred) ** 2))
        t2_root_mean_sq_error = torch.sqrt(t2_mean_sq_error)
        self.log(f"{data_type}_t2_mape", t2_mean_abs_perc_error.cpu())
        self.log(f"{data_type}_t2_mse", t2_mean_sq_error.cpu())
        self.log(f"{data_type}_t2_rmse", t2_root_mean_sq_error.cpu())

    def log(self, field: str, value):
        self._log[field].append(value)

    def on_epoch_end(self, epoch: int):
        if not os.path.exists(self._directory):
            os.mkdir(self._directory)

        values = [str(epoch)]
        for field, value in self._log.items():
            values.append(str(np.asarray(value).mean()))

        print(", ".join(['epoch', *self._log.keys()]))
        print(", ".join(values))

        with open(self.qualified_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if epoch == 1:  # First epoch. Write the csv header labels.
                writer.writerow(['epoch', *self._log.keys()])
            writer.writerow(values)

        self._log = defaultdict(list)
