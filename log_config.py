# log_config.py
import logging


def setup_logger():
    # Configure the root logger:
    logger = logging.getLogger()

    # Return early if handlers are already configured
    if logger.hasHandlers():
        return

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure the console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Configure a file handler, if you need logs to be written to a file:
    fh = logging.FileHandler("info.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
