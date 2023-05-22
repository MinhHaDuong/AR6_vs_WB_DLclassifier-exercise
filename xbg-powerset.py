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

from data import indicators_simulations, indicators_observations, get_data

AS_CHANGE=False

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

pairs = list(zip(indicators_simulations, indicators_observations))

for r in range(1, len(pairs) + 1):
    # Generate and print all subsets of size r
    for subset in itertools.combinations(pairs, r):
        isim, iobs = zip(*subset)
        data, labels = get_data(isim, iobs, as_change=AS_CHANGE)
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, random_state=42
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        score = classification_report(y_test, y_pred, output_dict=True)
        score["AUC-score"] = roc_auc_score(y_test, y_pred)
        result.loc[str(iobs)] = [score["AUC-score"], score["weighted avg"]["f1-score"]]
        print(result, "\n")

# %%
for v in indicators_observations:
    result[v] = [v in i for i in result.index]

result = result.reset_index().set_index(indicators_observations)

# In case we run the above twice
# result.drop(columns=indicators_observations, inplace=True)

# %%

result["Variables"] = result["Variables"].str.replace(
    "primary_energy_consumption", "tpec"
)
result = result.rename_axis(index={"primary_energy_consumption": "tpec"})

result["Variables"] = result["Variables"].str.replace("population", "pop")
result = result.rename_axis(index={"population": "pop"})

result["Variables"] = result["Variables"].str.replace(",)", ")")
result["Variables"] = result["Variables"].str.replace("'", "")

# %%

result.to_csv("result.csv")

# %%
result_byF1 = result.sort_values(by="F1", ascending=False)
result_byAUC = result.sort_values(by="AUC", ascending=False)

message = f"""
Are IPCC AR6 scenarios realistic?

Author: haduong@centre-cired.fr
Run saved: {datetime.datetime.now()}

We define a trajectory as a matrix with 6 columns and up to 4 rows.
The rows correspond to variables: CO2 emissions, GDP, populations, primary energy
The columns correspond to years, with a 5 years difference so that the trajectory is 25 years

The simulation trajectories are picked from the IPCC AR6 national scenario database
The observations trajectories are picked from owid-co2 dataset

We pool all the model, regions, years into two big sets of trajectories,
and trained a GBM classifier to distinguish simulations from observations.

Results shows that 
- Simulations are very distinguishable from observations.
- Trajectories with the 'population' variable are more distinguishable that those without

Conclusion so far:
    There is a bug in the 'Population' series.

{result}

Sorted by F1
{result_byF1.to_string(index=False)}

Sorted by AUC
{result_byAUC.to_string(index=False)}
"""

print(message)
with open("xbg-powerset.txt", "w", encoding="utf-8") as f:
    print(message, file=f)

# %%


def graph(score):
    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    y0 = result[score].min() * 0.99

    for variable, ax in zip(result.index.names, axes.flatten()):
        # Filter the series for when the variable is True and False
        true_values = result[score].xs(True, level=variable)
        false_values = result[score].xs(False, level=variable)
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
    plt.savefig(f"single_variable_{score}.png")
    plt.show()


graph("AUC")
graph("F1")
