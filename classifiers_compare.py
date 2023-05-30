""" Compare different machine learning algorithms to recognize simulations
Created on Mon May 29 17:24:47 2023

@author: haduong@centre-cired.fr
"""

import pandas as pd

from time import time, process_time
import datetime
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from keras.callbacks import EarlyStopping
from keras.models import Sequential

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from joblib import Parallel, delayed

from data import get_sets, all_vars
from classifier_others import model_dummy, model_lr, model_rf, model_svm, model_xgb
from classifier_mlp import model_mlp

# %%


def train_and_evaluate(model, x_train, x_test, y_train, y_test):
    if not hasattr(model, "fit"):
        raise TypeError("model must be a valid classifier with a fit method.")

    is_keras_model = isinstance(model, Sequential)

    start = process_time()

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

    train_t = process_time() - start

    print("Predicting, ", end="")
    start = process_time()

    y_pred = model.predict(x_test)
    if is_keras_model:
        y_pred_continuous = y_pred.ravel()
        y_pred = (y_pred_continuous > 0.5).astype(int)
    predict_t = process_time() - start

    print("Scoring.")
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


# %%


def compare(models_dict, x_train, x_test, y_train, y_test, parallelize=False):
    print("\nComparing classification models.")
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
            print(label, end=":   ")
            values = train_and_evaluate(model, x_train, x_test, y_train, y_test)
            return label, values

        parallel_results = Parallel(n_jobs=-1)(
            delayed(process_model)(label, model) for label, model in models_dict.items()
        )

        for label, values in parallel_results:
            result.loc[label] = values
    else:
        for label, model in models_dict.items():
            print(label, end=":   ")
            values = train_and_evaluate(model, x_train, x_test, y_train, y_test)
            result.loc[label] = values

    duration = round(time() - start_time)
    return result, duration


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
    "Multilayer perceptron 256/0.3/128/0.2/64/0.1": model_mlp(
        x_train.shape[1], 256, 0.3, 128, 0.2, 64, 0.1
    ),
    "Multilayer perceptron 256/0/128/0": model_mlp(x_train.shape[1], 256, 0, 128, 0),
    "Multilayer perceptron 256/0/128/0/64/0": model_mlp(
        x_train.shape[1], 256, 0, 128, 0, 64, 0
    ),
    "Multilayer perceptron 128/0/32/0.1": model_mlp(x_train.shape[1], 128, 0, 32, 0.1),
    "Multilayer perceptron 128/0/32/0": model_mlp(x_train.shape[1], 128, 0, 32, 0),
    "Multilayer perceptron 64/0.3/16/0.1/8/0": model_mlp(
        x_train.shape[1], 64, 0.3, 16, 0.1, 8, 0
    ),
    "Multilayer perceptron 64/0/16/0.1/8/0": model_mlp(
        x_train.shape[1], 64, 0, 16, 0.1, 8, 0
    ),
    "Multilayer perceptron 64/0/32/0/16/0": model_mlp(
        x_train.shape[1], 64, 0, 32, 0, 16, 0
    ),
    "Multilayer perceptron 32/0/16/0/8/0": model_mlp(
        x_train.shape[1], 32, 0, 16, 0, 8, 0
    ),
}


results = pd.DataFrame(columns=["result", "duration"])

results.loc["unscaled"] = compare(
    models_dict, x_train, x_test, y_train, y_test, parallelize=False
)
results.loc["parallel_unscaled"] = compare(
    models_dict, x_train, x_test, y_train, y_test, parallelize=True
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

results.loc["scaled"] = compare(
    models_dict, x_train_scaled, x_test_scaled, y_train, y_test, parallelize=False
)
results.loc["parallel_scaled"] = compare(
    models_dict, x_train_scaled, x_test_scaled, y_train, y_test, parallelize=True
)

# Resampled includes scaled
smote = SMOTE(random_state=0)
x_train_scaled_resampled, y_train_resampled = smote.fit_resample(
    x_train_scaled, y_train
)

results.loc["resampled"] = compare(
    models_dict,
    x_train_scaled_resampled,
    x_test_scaled,
    y_train_resampled,
    y_test,
    parallelize=False,
)
results.loc["parallel_resampled"] = compare(
    models_dict,
    x_train_scaled_resampled,
    x_test_scaled,
    y_train_resampled,
    y_test,
    parallelize=True,
)


# %% Print and save the profiled results

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 10000)

table_unscaled = results.loc["unscaled", "result"][
    results.loc["unscaled", "result"].columns[:-1]
].round(3)
table_parallel_unscaled = results.loc["parallel_unscaled", "result"][
    results.loc["parallel_unscaled", "result"].columns[:-1]
].round(3)

table_scaled = results.loc["scaled", "result"][
    results.loc["scaled", "result"].columns[:-1]
].round(3)
table_parallel_scaled = results.loc["parallel_scaled", "result"][
    results.loc["parallel_scaled", "result"].columns[:-1]
].round(3)

table_resampled = results.loc["resampled", "result"][
    results.loc["resampled", "result"].columns[:-1]
].round(3)
table_parallel_resampled = results.loc["parallel_resampled", "result"][
    results.loc["parallel_resampled", "result"].columns[:-1]
].round(3)


def message(formatter, **kwargs):
    return f"""
    Performance of different classifiers on the owid vs. ar6 dataset
    
    Author: haduong@centre-cired.fr
    Run saved: {datetime.datetime.now()}
    
    Unscaled (Duration: {results.loc['unscaled', 'duration']} seconds):
    {getattr(table_unscaled, formatter)(**kwargs)}
    
    Parallel Unscaled (Duration: {results.loc['parallel_unscaled', 'duration']} seconds):
    {getattr(table_parallel_unscaled, formatter)(**kwargs)}
    
    Scaled (Duration: {results.loc['scaled', 'duration']} seconds):
    {getattr(table_scaled, formatter)(**kwargs)}
    
    Parallel Scaled (Duration: {results.loc['parallel_scaled', 'duration']} seconds):
    {getattr(table_parallel_scaled, formatter)(**kwargs)}
    
    Resampled Scaled (Duration: {results.loc['resampled', 'duration']} seconds):
    {getattr(table_resampled, formatter)(**kwargs)}
    
    Parallel Resampled Scaled (Duration: {results.loc['parallel_resampled', 'duration']} seconds):
    {getattr(table_parallel_resampled, formatter)(**kwargs)}
    """


print(message("to_string"))

with open("classifiers_compare.txt", "w", encoding="utf-8") as f:
    print(message("to_csv", sep="\t"), file=f)

results.to_pickle("classifiers_compare.pkl")
