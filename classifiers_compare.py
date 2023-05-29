""" Compare different machine learning algorithms to recognize simulations
Created on Mon May 29 17:24:47 2023

@author: haduong@centre-cired.fr
"""

import pandas as pd

from time import time
import datetime
from sklearn.metrics import classification_report, roc_auc_score
from keras.callbacks import EarlyStopping
from keras.models import Sequential

from data import get_sets, all_vars
from classifier_others import model_dummy, model_lr, model_rf, model_svm, model_xgb
from classifier_mlp import model_mlp

# %%


def train_and_evaluate(model, x_train, x_test, y_train, y_test):
    if not hasattr(model, "fit"):
        raise TypeError("model must be a valid classifier with a fit method.")

    is_keras_model = isinstance(model, Sequential)

    start = time()

    if is_keras_model:
        print("Training Keras model.")
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
        print("Fitting, ", end="")
        model.fit(x_train, y_train)

    train_time = time() - start

    print("Predicting, ", end="")
    start = time()

    y_pred = model.predict(x_test)
    if is_keras_model:
        y_pred_continuous = y_pred.ravel()
        y_pred = (y_pred_continuous > 0.5).astype(int)
    predict_time = time() - start

    print("Scoring.")
    score = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
    f1 = score["weighted avg"]["f1-score"]
    precision = score["weighted avg"]["precision"]
    recall = score["weighted avg"]["recall"]
    accuracy = score["accuracy"]

    if is_keras_model:
        auc = roc_auc_score(y_test, y_pred_continuous)
    else:
        auc = roc_auc_score(y_test, y_pred)

    return train_time, predict_time, auc, f1, precision, recall, accuracy, model


def compare(models_dict, x_train, x_test, y_train, y_test):
    print("\nComparing classification models.")
    result = pd.DataFrame(
        columns=[
            "Train",
            "Predict",
            "AUC",
            "F1",
            "precision",
            "recall",
            "accuracy",
            "model",
        ]
    )
    result.index.name = "classifier"

    for label, model in models_dict.items():
        print(label, end=":   ")
        values = train_and_evaluate(model, x_train, x_test, y_train, y_test)
        result.loc[label] = values

    return result


# %% Run the comparison

x_train, x_test, y_train, y_test = get_sets(all_vars, as_change=True)

models_dict = {
    "Dummy baseline": model_dummy,
    "Logistic regression": model_lr,
    "Random forest": model_rf,
    "Support vector machine": model_svm,
    "Gradient boosting machine": model_xgb,
    "Multilayer perceptron": model_mlp(x_train.shape[1]),
}

result = compare(models_dict, x_train, x_test, y_train, y_test)


# %% Pretty printing

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 10000)
table = result[result.columns[:-1]].round(3)
print(table)

tab = "\t"
message = f"""
Performance of different classifiers on the owid vs. ar6 dataset

Author: haduong@centre-cired.fr
Run saved: {datetime.datetime.now()}

{table.to_csv(sep=tab)}
"""

with open("classifiers_compare.txt", "w", encoding="utf-8") as f:
    print(message, file=f)
