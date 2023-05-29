""" Compare different machine learning algorithms to recognize simulations
Created on Mon May 29 17:24:47 2023

@author: haduong@centre-cired.fr
"""

import pandas as pd

from time import time
import datetime
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from keras.callbacks import EarlyStopping
from keras.models import Sequential

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# from joblib import Parallel, delayed

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

    train_t = time() - start

    print("Predicting, ", end="")
    start = time()

    y_pred = model.predict(x_test)
    if is_keras_model:
        y_pred_continuous = y_pred.ravel()
        y_pred = (y_pred_continuous > 0.5).astype(int)
    predict_t = time() - start

    print("Scoring.")
    score = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
    f1 = score["weighted avg"]["f1-score"]
    precision = score["weighted avg"]["precision"]
    recall = score["weighted avg"]["recall"]
    # Imbalanced classes, disregard accuracy

    if is_keras_model:
        auc = roc_auc_score(y_test, y_pred_continuous)
    else:
        auc = roc_auc_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    tp = cm[1, 1]  # True positives
    tn = cm[0, 0]  # True negatives
    fp = cm[0, 1]  # False positives
    fn = cm[1, 0]  # False negatives

    return train_t, predict_t, auc, f1, precision, recall, tp, tn, fp, fn, model


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
            "tp",
            "tn",
            "fp",
            "fn",
            "model",
        ]
    )
    result.index.name = "classifier"

    for label, model in models_dict.items():
        print(label, end=":   ")
        values = train_and_evaluate(model, x_train, x_test, y_train, y_test)
        result.loc[label] = values

    # def process_model(label, model):
    #     print(label, end=":   ")
    #     values = train_and_evaluate(model, x_train, x_test, y_train, y_test)
    #     return label, values

    # parallel_results = Parallel(n_jobs=-1)(delayed(process_model)(label, model) for label, model in models_dict.items())

    # for label, values in parallel_results:
    #     result.loc[label] = values


    return result


# %% Run the comparison

x_train, x_test, y_train, y_test = get_sets(all_vars, as_change=True)

models_dict = {
    "Dummy baseline": model_dummy,
    "Logistic regression": model_lr,
    "Support vector machine": model_svm,
    "Random forest": model_rf,
    "Gradient boosting machine": model_xgb,
    "Multilayer perceptron 224/0.3/64/0.1": model_mlp(x_train.shape[1]),
    "Multilayer perceptron bis": model_mlp(x_train.shape[1]),
    "Multilayer perceptron ter": model_mlp(x_train.shape[1]),
    "Multilayer perceptron 256/0.3/128/0.2/64/0.1": model_mlp(x_train.shape[1], 256, 0.3, 128, 0.2, 64, 0.1),
    "Multilayer perceptron 256/0/128/0": model_mlp(x_train.shape[1], 256, 0, 128, 0),
    "Multilayer perceptron 256/0/128/0/64/0": model_mlp(x_train.shape[1], 256, 0, 128, 0, 64, 0),
#    "Multilayer perceptron 128/0.3/32/0.1": model_mlp(x_train.shape[1], 128, 0.3, 32, 0.1),
    "Multilayer perceptron 128/0/32/0.1": model_mlp(x_train.shape[1], 128, 0, 32, 0.1),
    "Multilayer perceptron 128/0/32/0": model_mlp(x_train.shape[1], 128, 0, 32, 0),
    "Multilayer perceptron 64/0.3/16/0.1/8/0": model_mlp(x_train.shape[1], 64, 0.3, 16, 0.1, 8, 0),
    "Multilayer perceptron 64/0/16/0.1/8/0": model_mlp(x_train.shape[1], 64, 0, 16, 0.1, 8, 0),
    "Multilayer perceptron 64/0/32/0/16/0": model_mlp(x_train.shape[1], 64, 0, 32, 0, 16, 0),
    "Multilayer perceptron 32/0/16/0/8/0": model_mlp(x_train.shape[1], 32, 0, 16, 0, 8, 0),
}

result_unscaled = compare(models_dict, x_train, x_test, y_train, y_test)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

result_scaled = compare(models_dict, x_train_scaled, x_test_scaled, y_train, y_test)

smote = SMOTE(random_state=0)
x_train_scaled_resampled, y_train_resampled = smote.fit_resample(x_train_scaled, y_train)
result_scaled_resampled = compare(models_dict, x_train_scaled_resampled, x_test_scaled, y_train_resampled, y_test)


# %% Pretty printing

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 10000)

table_unscaled = result_unscaled[result_unscaled.columns[:-1]].round(3)
print("\nUnscaled data", table_unscaled)
table_scaled = result_scaled[result_scaled.columns[:-1]].round(3)
print("\nScaled data", table_scaled)
table_scaled_resampled = result_scaled_resampled[result_scaled_resampled.columns[:-1]].round(3)
print("\nScaled resampled data", table_scaled_resampled)

tab = "\t"
message = f"""
Performance of different classifiers on the owid vs. ar6 dataset

Author: haduong@centre-cired.fr
Run saved: {datetime.datetime.now()}

Unscaled:
    
{table_unscaled.to_csv(sep=tab)}

Scaled:
    
{table_scaled.to_csv(sep=tab)}

Scaled and resampled:
    
{table_scaled_resampled.to_csv(sep=tab)}
"""

with open("classifiers_compare.txt", "w", encoding="utf-8") as f:
    print(message, file=f)
