import csv
import logging
import os
from collections import defaultdict

import torch

logger = logging.getLogger('mrf')


class DataLogger:
    """
    The idea is to call the log method each iteration of an epoch during both train and validation.
    The value is appended to a list. Then, when on_epoch_end() method is called. It will take the mean of
    all values in the list and save it to a csv file.
    """
    def __init__(self, directory: str):
        self._log = defaultdict(lambda: 0)
        self._directory = directory
        self._iterations = defaultdict(lambda: 0)

    @property
    def directory(self) -> str:
        return self._directory

    @property
    def filename(self) -> str:
        return "logs.csv"

    @property
    def qualified_filename(self) -> str:
        return f"{self.directory}/{self.filename}"

    def _check_dirs_exist(self):
        if not os.path.exists(self.directory):
            os.mkdir(self._directory)

    def log_error(self, predicted, labels, loss, data_type: str):
        batch_size = predicted.shape[0]
        mean_abs_perc_error = torch.mean(torch.abs(((labels - predicted) / labels))) * 100
        mean_sq_error = torch.mean(((labels - predicted) ** 2))
        root_mean_sq_error = torch.sqrt(mean_sq_error)
        mean_abs_error = torch.abs(labels - predicted).sum() / batch_size

        t1_true, t2_true = torch.transpose(labels, 0, 1)
        t1_pred, t2_pred = torch.transpose(predicted, 0, 1)

        t1_mean_abs_perc_error = torch.mean(torch.abs(((t1_true - t1_pred) / t1_true))) * 100
        t1_mean_sq_error = torch.mean(((t1_true - t1_pred) ** 2))
        t1_root_mean_sq_error = torch.sqrt(t1_mean_sq_error)
        t1_mean_abs_error = torch.abs(t1_true - t1_pred).sum() / batch_size

        t2_mean_abs_perc_error = torch.mean(torch.abs(((t2_true - t2_pred) / t2_true))) * 100
        t2_mean_sq_error = torch.mean(((t2_true - t2_pred) ** 2))
        t2_root_mean_sq_error = torch.sqrt(t2_mean_sq_error)
        t2_mean_abs_error = torch.abs(t2_true - t2_pred).sum() / batch_size

        if loss is not None:
            self.log(f"{data_type}_loss", (loss / len(labels)))

        self.log(f"{data_type}_mape", mean_abs_perc_error)
        self.log(f"{data_type}_t1_mape", t1_mean_abs_perc_error)
        self.log(f"{data_type}_t2_mape", t2_mean_abs_perc_error)

        self.log(f"{data_type}_mse", mean_sq_error)
        self.log(f"{data_type}_t1_mse", t1_mean_sq_error)
        self.log(f"{data_type}_t2_mse", t2_mean_sq_error)

        self.log(f"{data_type}_rmse", root_mean_sq_error)
        self.log(f"{data_type}_t1_rmse", t1_root_mean_sq_error)
        self.log(f"{data_type}_t2_rmse", t2_root_mean_sq_error)

        self.log(f"{data_type}_mae", mean_abs_error)
        self.log(f"{data_type}_t1_mae", t1_mean_abs_error)
        self.log(f"{data_type}_t2_mae", t2_mean_abs_error)

    def log(self, field: str, value):
        self._log[field] += value
        self._iterations[field] += 1

    def on_epoch_end(self, epoch: int):
        self._check_dirs_exist()

        values = [str(epoch)]
        for field, value in self._log.items():
            values.append(str('%.8f' % (value / self._iterations[field])))

        logger.info(", ".join(['epoch', *self._log.keys()]))
        logger.info(", ".join(values))

        with open(self.qualified_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if epoch == 1:  # First epoch. Write the csv header labels.
                writer.writerow(['epoch', *self._log.keys()])
            writer.writerow(values)

        self._log = defaultdict(lambda: 0)
        self._iterations = defaultdict(lambda: 0)
