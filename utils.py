"""Common functions and decorators

haduong@centre-cired.fr    2023-06-01
"""

import os
import pickle
import logging


def cache(module_name):
    """
    A decorator to load the result of a function from a file if it exists or
    compute it and save to the file if it doesn't exist.

    Use as    @cache(__file__)   so that the cache file name is the
    same as the module name with the extension changed to .pkl.

    Args:
        module_name (str): The name of the module (or script) that defines the function
            the decorator is decorating.

    Returns:
        function: The decorated function.
    """
    dir_name, file_name = os.path.split(module_name)
    file_base_name, _ = os.path.splitext(file_name)
    filename = os.path.join(dir_name, "cache", file_base_name + ".pkl")

    def real_decorator(function):
        def wrapper(*args, **kwargs):
            try:
                with open(filename, "rb") as file:
                    result = pickle.load(file)
                logging.info("   Success read file %s", filename)
            except (IOError, EOFError, pickle.UnpicklingError) as e_read:
                logging.info("Unable to fetch %s : %s.", filename, e_read)
                logging.info("Attempting now to create it.")
                try:
                    result = function(*args, **kwargs)
                    with open(filename, "wb") as file:
                        pickle.dump(result, file)
                    logging.info("Saved %s.", filename)
                except Exception as e_write:
                    logging.error(
                        "   An error occurred while saving to %s: %s.",
                        filename,
                        e_write,
                    )
                    return None
            return result

        return wrapper

    return real_decorator
