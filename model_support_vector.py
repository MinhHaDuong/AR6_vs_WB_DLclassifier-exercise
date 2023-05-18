#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:20:30 2023

@author: haduong
"""

from sklearn import svm
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from data import get_data

# %%

data, labels = get_data()

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)


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

"""
23115 instances of simulations variables ['Emissions|CO2', 'GDP|MER', 'Population', 'Primary Energy']
3018 instances of observation variables ['co2', 'gdp', 'population', 'primary_energy_consumption']
              precision    recall  f1-score   support

         0.0       0.99      0.97      0.98      4633
         1.0       0.78      0.95      0.86       594

    accuracy                           0.96      5227
   macro avg       0.89      0.96      0.92      5227
weighted avg       0.97      0.96      0.97      5227

AUC score: 0.9594873114191051
"""

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
