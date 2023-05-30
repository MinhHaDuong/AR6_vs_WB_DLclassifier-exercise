""" Compare different machine learning algorithms to recognize simulations
Created on Tue May 30 12:42:04 2023

@author: haduong
"""

import pandas as pd

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from classifier_others import model_dummy, model_lr, model_rf, model_svm, model_xgb
from classifier_mlp import model_mlp
from classifiers_compare import compare, pretty_print

from data import get_sets, all_vars


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

keys = [
    "unscaled",
    "parallel_unscaled",
    "scaled",
    "parallel_scaled",
    "resampled",
    "parallel_resampled",
]

print(pretty_print(results, keys, "to_string"))

with open("classifiers_compare_kind.csv", "w", encoding="utf-8") as f:
    print(pretty_print(results, keys, "to_csv", sep="\t"), file=f)

results.to_pickle("classifiers_compare_kind.pkl")
