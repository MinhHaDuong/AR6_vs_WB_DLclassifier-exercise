#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:34:14 2023

@author: haduong@centre-cired.fr
"""

import datetime
import itertools
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from data import get_sets, all_vars

# %%


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    # Normalize data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Rebalance data
    sm = SMOTE(random_state=42)
    x_train, y_train = sm.fit_resample(x_train, y_train)

    # After resampling, we may need to shuffle the data
    x_train, y_train = shuffle(x_train, y_train, random_state=42)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    score = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_pred)
    f1 = score["weighted avg"]["f1-score"]
    precision = score["weighted avg"]["precision"]
    recall = score["weighted avg"]["recall"]
    accuracy = score["accuracy"]

    # Calculate sample balance
    num_positives = sum(y_train)
    num_negatives = len(y_train) - num_positives
    sample_balance = num_positives / num_negatives

    return auc, f1, precision, recall, accuracy, sample_balance


def train_eval_powerset(model):
    result = pd.DataFrame(
        columns=["AUC", "F1", "Precision", "Recall", "Accuracy", "Sample Balance"]
    )
    result.index.name = "variables"

    for r in range(1, len(all_vars) + 1):
        for subset in itertools.combinations(all_vars, r):
            x_train, x_test, y_train, y_test = get_sets(subset, as_change=True)
            values = train_and_evaluate_model(model, x_train, y_train, x_test, y_test)
            key = str(subset).replace(",)", ")").replace("'", "")
            result.loc[key] = values

    return result


# %% Define the GBM classifier

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

# %% Run the classifications

result = train_eval_powerset(model)

# %%


TAB = "\t"

message = f"""
Are IPCC AR6 scenarios realistic?

Author: haduong@centre-cired.fr
Run saved: {datetime.datetime.now()}

We define a sequence as a matrix with 6 columns and up to 4 rows.
The rows correspond to variables: CO2 emissions, GDP, populations, primary energy
The columns correspond to years, with a 5 years difference so that the trajectory is 25 years

The simulation sequences are picked from the IPCC AR6 national scenario database
The observations sequences are picked from owid-co2 dataset

We pool all the model, regions, years into two big sets of sequences,
and trained a GBM classifier to distinguish simulations from observations.

Results shows that
- Simulations are quite distinguishable from observations, even when looking at series in difference.
- Sequences with the 'tpec' variable are less distinguishable.
- Sequences with the 'population' variable are more distinguishable.
-> Simulations are realistic for energy, but the demographic dynamics is questionable.


{result}

Cut and paste-ready: Sorted by F1, tab-separated
{result.sort_values(by="F1", ascending=False).round(3).to_csv(sep=TAB)}
"""

print(message)
with open("xbg-powerset.txt", "w", encoding="utf-8") as f:
    print(message, file=f)

# %%


def graph(score_name):
    plt.clf()
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    y0 = result[score_name].min() * 0.99

    for variable, ax in zip(["co2", "gdp", "pop", "tpec"], axes.flatten()):
        mask = result.index.str.contains(variable)
        true_values = result[mask][score_name]
        false_values = result[~mask][score_name]
        values = [true_values, false_values]

        ax.boxplot(values, positions=[1, 2])
        ax.set_xticklabels([f"{variable} present\n", f"{variable} absent\n"])
        ax.tick_params(axis="x", which="both", length=0)

        for i, value in enumerate(values, start=1):
            y = value
            x = [i] * len(y)
            ax.plot(x, y, "r.", alpha=0.8)

        ax.set_ylim(y0, 1)

    plt.suptitle(
        f"Effect of the presence of a variable on the classifier performance ({score_name} score)"
    )
    plt.tight_layout()
    plt.savefig(f"single_variable_{score_name}.png")
    plt.show()


graph("AUC")
graph("F1")
