import logging
import sys
from huri.core.file_sys import workdir
from time import strftime

exe_log_path = workdir / "exe"
if not exe_log_path.exists():
    exe_log_path.mkdir()


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s,%(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


exe_logger = None
if exe_logger is None:
    exe_logger = setup_logger("exe_logger", log_file=exe_log_path / f"{strftime('%Y%m%d-%H%M%S')}.log")
