import logging
import os
import sys

if os.path.exists("autowoe.log"):
    os.remove("autowoe.log")


def get_logger(name):
    """

    Returns:

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("autowoe.log", mode='a')
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(f"[%(name)s] [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
