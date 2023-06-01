""" Compare different ML classifiers observations/simulations.

Results are cached in .pkl for further processing.

Created on Tue May 30 12:42:04 2023

@author: haduong
"""

import pandas as pd
import pickle
import logging

from classifier_others import model_dummy, model_lr, model_rf, model_svm, model_xgb
from classifier_mlp import model_mlp
from classifiers_compare import compare

from data import get_sets
from log_config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

FILENAME="classifiers_compare_kind.pkl"

def get_results(filename):
    # Compare our default with the top four architectures from the model tuner
    x_train, _, _, _ = get_sets(diff=True)
    dim = x_train.shape[1]
    models_dict = {
        "Dummy baseline": model_dummy,
        "Logistic regression": model_lr,
        "Support vector machine": model_svm,
        "Random forest": model_rf,
        "Gradient boosting machine": model_xgb,
        "Multilayer perceptron default": model_mlp(dim),
        "Multilayer perceptron bis": model_mlp(dim),
        "Multilayer perceptron ter": model_mlp(dim),
        "Multilayer perceptron 128/0/48/0.1/16/0": model_mlp(dim, 128, 0, 48, 0.1, 16, 0),
        "Multilayer perceptron 96/0/48/0.2/8/0.2": model_mlp(dim, 96, 0, 48, 0.2, 8, 0.2),
        "Multilayer perceptron 64/0/32/0/32/0.1": model_mlp(dim, 64, 0, 32, 0, 32, 0.1),
        "Multilayer perceptron 32/0.1/32/0.1/32/0.2": model_mlp(
            dim, 32, 0.1, 32, 0.1, 32, 0.2
        ),
    }

    results = pd.DataFrame(columns=["result", "duration"])

    def run(case, normalize, rebalance):
        x_train, x_test, y_train, y_test = get_sets(
            diff=True, normalize=normalize, rebalance=rebalance
        )
        results.loc[case] = compare(
            models_dict, x_train, x_test, y_train, y_test, parallelize=False
        )
        results.loc["parallel " + case] = compare(
            models_dict, x_train, x_test, y_train, y_test, parallelize=True
        )

    run("raw", False, False)
    run("normalized", True, False)
    run("balanced", False, True)
    run("base", True, True)

    return results


try:
    results = pd.read_pickle(FILENAME)
    logging.info(f"Success read  file {FILENAME} ")
except (IOError, EOFError, pickle.UnpicklingError) as e_read:
    logging.info(f"Unable to access {FILENAME} : {e_read}")
    logging.info("Attempting to create it.")
    try:
        results = get_results()
        results.to_pickle(FILENAME)
        logging.info("Classifiers comparison saved successfully!")
    except Exception as e:
        logging.error(f"An error occurred while saving the classifiers comparison: {e}")
