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

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

import matplotlib.pyplot as plt

from data import get_data, all_vars

AS_CHANGE = True

# %%

# Define the parameters for the GBM classifier
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

# Initialize the GBM classifier
model = xgb.XGBClassifier(**params)

# %%

result = pd.DataFrame(columns=["AUC", "F1"])
result.index.name = "Variables"

for r in range(1, len(all_vars) + 1):
    for subset in itertools.combinations(all_vars, r):
        data, labels = get_data(subset, as_change=AS_CHANGE)
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, random_state=42
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        score = classification_report(y_test, y_pred, output_dict=True)
        score["AUC-score"] = roc_auc_score(y_test, y_pred)
        result.loc[str(subset)] = [
            score["AUC-score"],
            score["weighted avg"]["f1-score"],
        ]
        print(result, "\n")

# %%

result.reset_index(level=0, inplace=True)
result["Variables"] = result["Variables"].str.replace(",)", ")")
result["Variables"] = result["Variables"].str.replace("'", "")
result.set_index("Variables", inplace=True)

result_byF1 = result.sort_values(by="F1", ascending=False)
result_byAUC = result.sort_values(by="AUC", ascending=False)

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
- Simulations are very distinguishable from observations when looking at levels.
- Simulations are less distinguishable from observations when looking at changes.
- Trajectories with the 'population' variable are more distinguishable that those without

{result}

Sorted by F1
{result_byF1.to_string()}

Sorted by AUC
{result_byAUC.to_string()}
"""

print(message)
with open(f"xbg-powerset-change{AS_CHANGE}.txt", "w", encoding="utf-8") as f:
    print(message, file=f)

# %%


def clean(result):
    if "level_0" in result.columns:
        result = result.drop(columns="level_0")
    if "index" in result.columns:
        result = result.drop(columns="index")
    for v in list(all_vars):
        if v in result.columns:
            result = result.drop(columns=v)
    if "Variables" not in result.columns:
        variables = [
            subset
            for r in range(1, len(all_vars) + 1)
            for subset in itertools.combinations(all_vars, r)
        ]
        result["Variables"] = variables
    return result


result = clean(result)

try:
    result = result.reset_index()
except ValueError:
    print("Failed to reset_index. Any column names conflict with index levels?")

try:
    result = result.set_index("Variables")
except KeyError:
    print("Failed to set_index. Is Variable column present?")

for v in list(all_vars):
    print(f"Checking if we have column {v}")
    if v not in result.columns:
        print("... Creating it")
        result[v] = [v in i for i in result.index]

result = result.reset_index()  # Move "Variables" in the columns
result.set_index(list(all_vars), inplace=True)

clean(result)

# %%


def graph(score_name):
    plt.clf()
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    y0 = result[score_name].min() * 0.99

    for variable, ax in zip(result.index.names, axes.flatten()):
        # Filter the series for when the variable is True and False
        true_values = result[score_name].xs(True, level=variable)
        false_values = result[score_name].xs(False, level=variable)
        values = [true_values, false_values]

        ax.boxplot(values, labels=["In", "·Out"], positions=[1, 2])
        # Add individual data points
        for i, value in enumerate(values, start=1):
            y = value
            x = [i] * len(y)
            ax.plot(x, y, "r.", alpha=0.8)  # 'r.' specifies red dots

        ax.set_ylim(y0, 1)
        ax.set_title(f"{variable}")
        ax.set_ylabel(score)

    plt.tight_layout()
    plt.savefig(f"single_variable_{score_name}.png")
    plt.show()


graph("AUC")
graph("F1")
