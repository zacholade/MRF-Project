import configparser


config = configparser.ConfigParser()
config.read("config.ini")


class Configuration:
    """
    A class to effectively parse the config.ini file.
    """
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
    def rnn_input_size(self) -> int:
        return self._config.getint(f"{self.network_name}Hyperparameters", "rnn_input_size")

    @property
    def rnn_hidden_size(self) -> int:
        return self._config.getint(f"{self.network_name}Hyperparameters", "rnn_hidden_size")

    @property
    def rnn_num_layers(self) -> int:
        return self._config.getint(f"{self.network_name}Hyperparameters", "rnn_num_layers")

    @property
    def rnn_bidirectional(self) -> bool:
        return self._config.getboolean(f"{self.network_name}Hyperparameters", "rnn_bidirectional")

    @property
    def spatial_pooling(self) -> str:
        return self._config.get(f"{self.network_name}Hyperparameters", "spatial_pooling")

    @property
    def patch_size(self) -> int:
        return self._config.getint(f"{self.network_name}Hyperparameters", "patch_size")

    @property
    def factorise(self) -> bool:
        return self._config.getboolean(f"{self.network_name}Hyperparameters", "factorise")

    @property
    def use_attention(self) -> bool:
        return self._config.getboolean(f"{self.network_name}Hyperparameters", "use_attention")

    @property
    def cbam_attention(self) -> bool:
        return self._config.getboolean(f"{self.network_name}Hyperparameters", "cbam_attention")

    @property
    def rcab_attention(self) -> bool:
        return self._config.getboolean(f"{self.network_name}Hyperparameters", "rcab_attention")

    @property
    def num_temporal_features(self) -> int:
        return self._config.getint(f"{self.network_name}Hyperparameters", "num_temporal_features")

    @property
    def non_local_level(self) -> int:
        return self._config.getint(f"{self.network_name}Hyperparameters", "non_local_level")

    @property
    def modern(self) -> bool:
        return self._config.getboolean(f"{self.network_name}Hyperparameters", "modern")

    @property
    def dimensionality_reduction_level(self) -> int:
        return self._config.getint(f"{self.network_name}Hyperparameters", "dimensionality_reduction_level")

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
