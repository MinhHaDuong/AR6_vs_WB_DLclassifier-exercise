#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:20:30 2023

@author: haduong
"""

from sklearn import svm
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV

from data import get_sets

# %%

x_train, x_test, y_train, y_test = get_sets()


# Initialize an SVM model
# kernel, C, and gamma tuned from cell below
model = svm.SVC(kernel="rbf", random_state=42, C=100, gamma=1, class_weight="balanced")

# Train the model
model.fit(x_train, y_train)

# Use the model to make predictions on the test set
y_pred = model.predict(x_test)

# Print a classification report to see the model's performance
print(classification_report(y_test, y_pred))

# Also compute the AUC score
print("AUC score:", roc_auc_score(y_test, y_pred))

# %% Tune it

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["linear", "rbf"],
}

# Initialize an SVM model
model = svm.SVC(random_state=42)

# Initialize the GridSearch object
grid_search = GridSearchCV(model, param_grid, cv=5)

# Fit it to the data and find the best hyperparameters
grid_search.fit(x_train, y_train)

# Print the best parameters and the corresponding score
print("Best parameters: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# We can also use the best model found by GridSearchCV
best_model = grid_search.best_estimator_
