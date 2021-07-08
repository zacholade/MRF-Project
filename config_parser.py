import configparser
import typing


config = configparser.ConfigParser()
config.read("config.ini")


class Configuration:
    def __init__(self, network_name: str, filename: str, debug: bool):
        self.network_name = network_name
        self._config = configparser.ConfigParser()
        self._config.read(filename)

        self.debug = debug

    @property
    def total_epochs(self) -> int:
        return self._config.getint(f"{self.network_name}Hyperparameters", "total_epochs")

    @property
    def batch_size(self) -> int:
        return self._config.getint(f"{self.network_name}Hyperparameters", "batch_size")

    @property
    def lr(self) -> float:
        return self._config.getfloat(f"{self.network_name}Hyperparameters", "lr")

    @property
    def lr_step_size(self) -> int:
        return self._config.getint(f"{self.network_name}Hyperparameters", "lr_step_size")

    @property
    def lr_gamma(self) -> float:
        return self._config.getfloat(f"{self.network_name}Hyperparameters", "lr_gamma")

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
