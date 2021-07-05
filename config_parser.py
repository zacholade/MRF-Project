import configparser
import typing


config = configparser.ConfigParser()
config.read("config.ini")


class Configuration:
    def __init__(self, filename: str, debug: bool):
        self._config = configparser.ConfigParser()
        self._config.read(filename)

        self.debug = debug

    @property
    def total_epochs(self) -> int:
        return self._config.getint("TrainingHyperparameters", "total_epochs")

    @property
    def batch_size(self) -> int:
        return self._config.getint("TrainingHyperparameters", "batch_size")

    @property
    def learning_rate(self) -> float:
        return self._config.getfloat("TrainingHyperparameters", "learning_rate")

    @property
    def validate(self) -> bool:
        return self._config.getboolean("Settings", "validate")

    @property
    def limit_iterations(self) -> int:
        if not self.debug:
            return -1
        return self._config.getint("Debug", "limit_iterations")

    @property
    def limit_number_files(self) -> int:
        if not self.debug:
            return -1
        return self._config.getint("Debug", "limit_number_files")
