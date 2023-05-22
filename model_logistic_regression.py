#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:08:01 2023

@author: haduong
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV

from data import get_sets

# %%

x_train, x_test, y_train, y_test = get_sets()


# Initialize a Logistic Regression model
# Note: regularization strength (C) and penalty tuned from cell below
model = LogisticRegression(solver="liblinear", random_state=42, C=1000, penalty="l1")

# Train the model
model.fit(x_train, y_train)

# Use the model to make predictions on the test set
y_pred = model.predict(x_test)

# You can print a classification report to see the model's performance
print(classification_report(y_test, y_pred))

# You can also compute the AUC (Area Under the Curve) score
print("AUC score:", roc_auc_score(y_test, y_pred))

# %% Tune it

# Define the hyperparameter grid
param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "penalty": ["l1", "l2"],
}

# Initialize a Logistic Regression model
model = LogisticRegression(solver="liblinear", random_state=42)

# Initialize the GridSearch object
grid_search = GridSearchCV(model, param_grid, cv=5)

# Fit it to the data and find the best hyperparameters
grid_search.fit(x_train, y_train)

# Print the best parameters and the corresponding score
print("Best parameters: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# We can also use the best model found by GridSearchCV
best_model = grid_search.best_estimator_
