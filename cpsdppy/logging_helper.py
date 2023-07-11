# -*- coding: utf-8 -*-

"""Logging configuration helpers"""

import copy
import io
import logging
import logging.config
import os


def setup(dir: str = "log", prefix: str = "") -> None:
    """Set up loggers

    This set up log handlers and sybolic links to log files..

    1) `console` handler: Output log messages to console.
    2) `infofile`, `debugfile` handlers: Save log messages to files.
       By default, the log files are named as `log/info.txt` and
       `log/debug.txt` in the current directory.
    3) `log/info_link.txt`, `log/debug_link.txt`: Symbolic links to
       the files created by `infofile`, `debugfile` handlers.

    Parameters
    ----------
    dir : str, default 'log'
        Directory to save the files created by `infofile` and `debugfile`
        handlers.
    prefix : str, optional
        Prefix added to the files created by `infofile` and `debugfile`
        handlers.
    """
    config: dict = copy.deepcopy(default_config)
    if dir:
        os.makedirs(dir, exist_ok=True)
    if prefix:
        prefix = prefix + "_"
    if dir or prefix:
        for key in ["debugfile", "infofile"]:
            config["handlers"][key]["filename"] = os.path.join(
                dir,
                prefix + config["handlers"][key]["filename"],
            )
    else:
        for key in ["debugfile", "infofile"]:
            del config["handlers"][key]
            config["root"]["handlers"].remove(key)
    logging.config.dictConfig(config)

    os.makedirs("log", exist_ok=True)
    for key in ["debugfile", "infofile"]:
        if key not in config["handlers"]:
            continue
        symlink_path = os.path.join(
            "log", key.replace("file", "") + "_link.txt"
        )
        if os.path.exists(symlink_path):
            os.remove(symlink_path)
        os.symlink(
            os.path.abspath(config["handlers"][key]["filename"]),
            symlink_path,
        )


class save_log(object):
    """Capture log and write to a file or retrieve as a string

    This is a context manager which captures log messages.

    Examples
    --------
    >>> import logging
    >>> def foo():
    ...     logger = logging.getLogger()
    ...     logger.warning('this is a warning')
    >>> with save_log() as f:
    ...     foo()
    >>> print(f.get())
    this is a warning

    Note that by default log messages of level info is suppressed.
    We need to configure the logger to see the log messages of level info
    or lower:

    >>> import logging
    >>> def foo():
    ...     logger = logging.getLogger()
    ...     logger.info('hello world')
    ...     logger.warning('this is a warning')
    >>> logging.getLogger().setLevel(logging.INFO)
    >>> with save_log() as f:
    ...     foo()
    >>> print(f.get())
    hello world
    this is a warning

    One can capture log messages of a specific logger.

    >>> logging.getLogger().setLevel(logging.INFO)
    >>> def foo():
    ...     root_logger = logging.getLogger()
    ...     root_logger.info('root log')
    ...     child_logger = logging.getLogger('child.logger')
    ...     child_logger.info('child log')
    >>> with save_log(logger_name='child') as f:
    ...     foo()
    >>> print(f.get())
    child log

    """

    def __init__(
        self,
        file_path=None,
        logger_name=None,
        level=logging.INFO,
        propagate=None,
    ):
        """Initialise a save_log instance

        Parameters
        ----------
        file_path : str, optional
            If given, a file handler is created.
        logger_name : str, optional
            The name of the logger whose messages are to be saved.
        level : int, default logging.INFO
            The level of the log messages to be captured.
            Note that one needs to use `logger.setLevel` to process
            log messages whose levels are INFO or lower.
        propagate : bool, optional
            If given, `logger.propagate` attribute is set to be this value
            within the context.
        """
        self.string_io = io.StringIO()
        self._logger = logging.getLogger(logger_name)
        self.string_io_handler = logging.StreamHandler(self.string_io)
        self.string_io_handler.setLevel(level)
        self._logger.addHandler(self.string_io_handler)
        self.file_path = file_path
        if self.file_path is not None:
            formatter = logging.Formatter("%(asctime)s | %(message)s")
            self.file_handler = logging.FileHandler(self.file_path)
            self.file_handler.setLevel(level)
            self.file_handler.setFormatter(formatter)
            self._logger.addHandler(self.file_handler)
        if propagate is None:
            self._original_propagate = None
        else:
            self._original_propagate = self._logger.propagate
            self._logger.propagate = propagate

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self._logger.removeHandler(self.string_io_handler)
        if self.file_path:
            self._logger.removeHandler(self.file_handler)
        if self._original_propagate is not None:
            self._logger.propagate = self._original_propagate

    def get(self):
        return self.string_io.getvalue()


default_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "plain": {"class": "logging.Formatter", "format": "%(message)s"},
        "timed": {
            "class": "logging.Formatter",
            "format": "%(asctime)s | %(message)s",
        },
        "detailed": {
            "class": "logging.Formatter",
            "format": "%(asctime)s | %(levelname)-8s | %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "stream": "ext://sys.stdout",
            "formatter": "plain",
        },
        "debugfile": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "debug.txt",
            "maxBytes": 10000000,
            "backupCount": 3,
        },
        "infofile": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "detailed",
            "filename": "info.txt",
            "maxBytes": 10000000,
            "backupCount": 3,
        },
    },
    "loggers": {"matplotlib": {"level": "WARNING", "propagate": False}},
    "root": {
        "level": "DEBUG",
        "handlers": [
            "console",
            "debugfile",
            "infofile",
        ],
    },
}


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
