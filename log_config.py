"""Setup logging. Filter WARN from Tensorflow, pass on our own INFO

Created on Tue May  9 19:43:16 2023
@author: haduong@centre-cired.fr
"""
# log_config.py

import logging
import warnings


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
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Configure a file handler, if you need logs to be written to a file:
    file_handler = logging.FileHandler("info.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


warnings.filterwarnings(
    "ignore", category=UserWarning, module="tensorflow_addons.utils.tfa_eol_msg"
)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings(
    "ignore",
    message="Could not find cuda drivers on your machine, GPU will not be used.",
)
