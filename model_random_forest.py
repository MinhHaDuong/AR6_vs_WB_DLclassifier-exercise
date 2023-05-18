#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:50:29 2023

@author: haduong
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

from data import get_data

# %%

data, labels = get_data()

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Initialize a Random Forest classifier
model = RandomForestClassifier(
    bootstrap=False,
    max_depth=20,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=300,
    random_state=42,
)

# Train the model
model.fit(x_train, y_train)

# Use the model to make predictions on the test set
y_pred = model.predict(x_test)

# Print a classification report to see the model's performance
print(classification_report(y_test, y_pred))

# Also compute the AUC score
print("AUC score:", roc_auc_score(y_test, y_pred))


# %% Tune it

# Define the hyperparameter distribution
param_dist = {
    "n_estimators": stats.randint(100, 500),
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": stats.randint(2, 10),
    "min_samples_leaf": stats.randint(1, 4),
    "bootstrap": [True, False],
}

# Initialize a Random Forest classifier
model = RandomForestClassifier(random_state=42)

# Initialize the GridSearch object
random_search = RandomizedSearchCV(model, param_dist, cv=3, n_jobs=-1, verbose=2)

# Fit it to the data and find the best hyperparameters
random_search.fit(x_train, y_train)

# Print the best parameters and the corresponding score
print("Best parameters: ", random_search.best_params_)
print("Best cross-validation score: ", random_search.best_score_)

# We can also use the best model found by GridSearchCV
best_model = random_search.best_estimator_
