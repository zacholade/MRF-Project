import csv
import os
from collections import defaultdict

import numpy as np


class DataLogger:
    """
    The idea is to call the log method each iteration of an epoch during both train and validation.
    The value is appended to a list. Then, when on_epoch_end() method is called. It will take the mean of
    all values in the list and save it to a csv file.
    """
    def __init__(self, directory: str):
        self._log = defaultdict(list)
        self._directory = directory
        self._first_epoch = True

    @property
    def directory(self) -> str:
        return self._directory

    @property
    def filename(self) -> str:
        return f"logs.csv"

    @property
    def qualified_filename(self) -> str:
        return f"{self.directory}/{self.filename}"

    def log(self, field: str, value):
        self._log[field].append(value)

    def on_epoch_end(self, epoch: int):
        if not os.path.exists(self._directory):
            os.mkdir(self._directory)

        values = [str(epoch)]
        for field, value in self._log.items():
            values.append(str(np.asarray(value).mean()))

        with open(self.qualified_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if self._first_epoch:
                writer.writerow(['epoch', *self._log.keys()])
                self._first_epoch = not self._first_epoch
            writer.writerow(values)

        self._log = defaultdict(list)
