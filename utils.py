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
                with open(filename, "rb") as f:
                    result = pickle.load(f)
                logging.info(f"   Success read file {filename}")
            except (IOError, EOFError, pickle.UnpicklingError) as e_read:
                logging.info(f"Unable to fetch {filename} : {e_read} .")
                logging.info("Attempting now to create it.")
                try:
                    result = function(*args, **kwargs)
                    with open(filename, "wb") as f:
                        pickle.dump(result, f)
                    logging.info(f"Saved {filename}.")
                except Exception as e_write:
                    logging.error(
                        f"   An error occurred while saving to {filename}: {e_write} "
                    )
                    return None
            return result

        return wrapper

    return real_decorator
