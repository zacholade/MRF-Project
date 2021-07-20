import logging
import sys
from contextlib import contextmanager

ROOT_LOGGER_NAME = 'mrf'


class LoggingMixin:
    @property
    def logger(self) -> logging.Logger:
        name = '.'.join([ROOT_LOGGER_NAME, self.__module__, self.__class__.__name__])
        return logging.getLogger(name)


@contextmanager
def setup_logging(debug: bool) -> logging.Logger:
    logger = None
    try:
        default_logging_format = ''.join([
            "[",
            "%(asctime)s ",
            "%(levelname)-8s ",
            "%(filename)-15s:",
            "%(lineno)3s - ",
            "%(funcName)-10s ",
            "] ",
            "%(message)s",
        ])
        logger = logging.getLogger()
        logging.basicConfig(format=default_logging_format)
        logger.setLevel(logging.DEBUG) if debug else logger.setLevel(logging.INFO)
        yield logger
    finally:
        # __exit__
        if logger:
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
