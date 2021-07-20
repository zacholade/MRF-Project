import configparser
import typing


config = configparser.ConfigParser()
config.read("config.ini")


class Configuration:
    def __init__(self, network_name: str, filename: str, debug: bool):
        self.network_name = network_name
        self._config = configparser.ConfigParser(inline_comment_prefixes='#')
        self._config.read(filename)

        self.debug = debug

    @property
    def seq_len(self) -> int:
        return self._config.getint(f"{self.network_name}Hyperparameters", "seq_len")

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
    def gru(self) -> bool:
        return self._config.getboolean(f"{self.network_name}Hyperparameters", "gru")

    @property
    def lstm_input_size(self) -> int:
        return self._config.getint(f"{self.network_name}Hyperparameters", "lstm_input_size")

    @property
    def lstm_hidden_size(self) -> int:
        return self._config.getint(f"{self.network_name}Hyperparameters", "lstm_hidden_size")

    @property
    def lstm_num_layers(self) -> int:
        return self._config.getint(f"{self.network_name}Hyperparameters", "lstm_num_layers")

    @property
    def lstm_bidirectional(self) -> bool:
        return self._config.getboolean(f"{self.network_name}Hyperparameters", "lstm_bidirectional")

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
