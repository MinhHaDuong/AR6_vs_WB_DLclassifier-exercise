""" Autoencode IPAT sequences and see 

Created on Tue May 30 12:24:16 2023

@author: haduong@centre-cired.fr
"""

import pandas as pd

from data import get_sets, all_vars
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from classifier_others import model_dummy, model_xgb
from classifier_mlp import model_mlp

from classifiers_compare import compare, pretty_print


def perform_dimensionality_reduction(x_raw_train, x_raw_test):
    # Perform dimensionality reduction on raw data to obtain latent space data
    # Placeholder for now
    return x_raw_train, x_raw_test


# %% Run the comparison

x_raw_train, x_raw_test, y_train, y_test = get_sets(all_vars, as_change=True)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_raw_train)
x_test_scaled = scaler.transform(x_raw_test)

smote = SMOTE(random_state=0)
x_train_scaled_resampled, y_train_resampled = smote.fit_resample(
    x_train_scaled, y_train
)

models_dict = {
    "Dummy baseline": model_dummy,
    "Gradient boosting machine": model_xgb,
    "Multilayer perceptron 256/0/128/0": model_mlp(x_raw_train.shape[1], 256, 0, 128, 0),
}

results = pd.DataFrame(columns=["result", "duration"])

results.loc["raw"] = compare(models_dict, x_raw_train, x_raw_test, y_train, y_test)

results.loc["normalized"] = compare(models_dict,
                                    x_train_scaled_resampled,
                                    x_test_scaled,
                                    y_train_resampled,
                                    y_test)

x_latent_train, x_latent_test = perform_dimensionality_reduction(x_raw_train, x_raw_test)
results.loc["latent"] = compare(models_dict, x_latent_train, x_latent_test, y_train, y_test)

# %% Print and save the results

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 10000)

table_raw = results.loc["raw", "result"][
    results.loc["raw", "result"].columns[:-5]
].round(3)
table_normalized = results.loc["normalized", "result"][
    results.loc["normalized", "result"].columns[:-5]
].round(3)
table_latent = results.loc["latent", "result"][
    results.loc["latent", "result"].columns[:-5]
].round(3)

print(table_raw)
print()
print(table_normalized)
print()
print(table_latent)

# %%

keys = ["raw", "normalized", "latent"]

print(pretty_print(results, keys, "to_string"))