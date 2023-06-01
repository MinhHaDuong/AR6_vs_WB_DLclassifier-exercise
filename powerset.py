#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:34:14 2023

@author: haduong@centre-cired.fr
"""

import logging
import itertools
import pandas as pd
import xgboost as xgb

from sklearn.metrics import classification_report, roc_auc_score

from log_config import setup_logger
from utils import cache
from data import get_sets, all_vars

setup_logger()
logger = logging.getLogger(__name__)


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    score = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_pred)
    f1 = score["weighted avg"]["f1-score"]
    precision = score["weighted avg"]["precision"]
    recall = score["weighted avg"]["recall"]
    accuracy = score["accuracy"]

    return auc, f1, precision, recall, accuracy


def train_eval_powerset(model):
    logging.info("Run observation/simulation classifier for all subsets.")
    result = pd.DataFrame(columns=["AUC", "F1", "Precision", "Recall", "Accuracy"])
    result.index.name = "variables"

    for r in range(1, len(all_vars) + 1):
        for subset in itertools.combinations(all_vars, r):
            key = str(subset).replace(",)", ")").replace("'", "")
            logging.info(key)
            x_train, x_test, y_train, y_test = get_sets(
                subset, diff=True, normalize=True, rebalance=True
            )
            values = train_and_evaluate_model(model, x_train, y_train, x_test, y_test)
            result.loc[key] = values

    return result


@cache(__file__)
def get_results():
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.1,
        "max_depth": 7,
        "subsample": 0.889,
        "colsample_bytree": 0.657,
        "learning_rate": 0.205,
        "n_estimators": 360,
    }
    model = xgb.XGBClassifier(**params)
    results = train_eval_powerset(model)
    return results


# When run directly, create the .pkl if necessary
if __name__ == "__main__":
    get_results()
