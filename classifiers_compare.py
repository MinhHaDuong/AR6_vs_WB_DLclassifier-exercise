""" Functions to compare different machine learning classifiers and prettyprint result
Created on Mon May 29 17:24:47 2023

@author: haduong@centre-cired.fr
"""

import datetime
import logging
import pandas as pd

from time import time, process_time
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from keras.callbacks import EarlyStopping
from keras.models import Sequential

from joblib import Parallel, delayed

from log_config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


def train_and_evaluate(model, x_train, x_test, y_train, y_test):
    if not hasattr(model, "fit"):
        raise TypeError("model must be a valid classifier with a fit method.")

    is_keras_model = isinstance(model, Sequential)

    start = process_time()

    if is_keras_model:
        logging.info("Training Keras model.")
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=3, verbose=1, restore_best_weights=True
        )
        model.fit(
            x_train,
            y_train,
            batch_size=64,
            epochs=30,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping],
        )
    else:
        logging.info("Fitting ML classifier model.")
        model.fit(x_train, y_train)

    train_t = process_time() - start

    logging.info("Making predictions.")
    start = process_time()

    y_pred = model.predict(x_test)
    if is_keras_model:
        y_pred_continuous = y_pred.ravel()
        y_pred = (y_pred_continuous > 0.5).astype(int)
    predict_t = process_time() - start

    logging.info("Computing evaluation metrics.")
    cm = confusion_matrix(y_test, y_pred)
    tp = cm[1, 1]  # True positives
    tn = cm[0, 0]  # True negatives
    fp = cm[0, 1]  # False positives
    fn = cm[1, 0]  # False negatives

    score = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
    f1 = score["weighted avg"]["f1-score"]
    precision = score["weighted avg"]["precision"]
    recall = score["weighted avg"]["recall"]
    # Imbalanced classes, disregard accuracy

    if is_keras_model:
        auc = roc_auc_score(y_test, y_pred_continuous)
    else:
        auc = roc_auc_score(y_test, y_pred)

    return train_t, predict_t, tp, tn, fp, fn, auc, f1, precision, recall, model


def compare(models_dict, x_train, x_test, y_train, y_test, parallelize=True):
    logging.info("Comparing classification models.")
    result = pd.DataFrame(
        columns=[
            "Train",
            "Predict",
            "tp",
            "tn",
            "fp",
            "fn",
            "AUC",
            "F1",
            "precision",
            "recall",
            "model",
        ]
    )
    result.index.name = "classifier"

    start_time = time()

    if parallelize:

        def process_model(label, model):
            logging.info(label)
            values = train_and_evaluate(model, x_train, x_test, y_train, y_test)
            return label, values

        parallel_results = Parallel(n_jobs=-1)(
            delayed(process_model)(label, model) for label, model in models_dict.items()
        )

        for label, values in parallel_results:
            result.loc[label] = values
    else:
        for label, model in models_dict.items():
            logging.info(label)
            values = train_and_evaluate(model, x_train, x_test, y_train, y_test)
            result.loc[label] = values

    duration = round(time() - start_time)
    return result, duration


def format_table(results, key, formatter, **kwargs):
    table = results.loc[key, "result"][results.loc[key, "result"].columns[:-5]].round(3)
    duration = results.loc[key, "duration"]

    formatted_table = getattr(table, formatter)(**kwargs)

    return (
        f"\n{key.capitalize()} (Duration: {duration} seconds):\n"
        + f"{formatted_table}\n"
    )


def pretty_print(results, keys, formatter, **kwargs):
    message = "Performance of different classifiers on the owid vs. ar6 dataset\n"
    message += "Author: haduong@centre-cired.fr\n"
    message += f"Run saved: {datetime.datetime.now()}\n"

    for key in keys:
        message += format_table(results, key, formatter, **kwargs)
    return message
