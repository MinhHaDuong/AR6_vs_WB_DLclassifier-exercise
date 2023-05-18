#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:20:59 2023

@author: haduong
"""

import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

from data import get_data

# %%

data, labels = get_data()

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)


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

# Train the model
model.fit(x_train, y_train)

# Use the model to make predictions on the test set
y_pred = model.predict(x_test)

# Print a classification report to see the model's performance
print(classification_report(y_test, y_pred))

# Also compute the AUC score
print("AUC score:", roc_auc_score(y_test, y_pred))

# %%
"""
             precision    recall  f1-score   support

         0.0       1.00      1.00      1.00      4633
         1.0       1.00      0.99      1.00       594

    accuracy                           1.00      5227
   macro avg       1.00      1.00      1.00      5227
weighted avg       1.00      1.00      1.00      5227

AUC score: 0.9966329966329966
"""

# %% Tune it


# Define the parameter distributions for random search
param_dist = {
    "learning_rate": uniform(0.01, 0.3),
    "max_depth": randint(3, 10),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "n_estimators": randint(100, 500),
}

# Initialize the GBM classifier
model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")

# Initialize the RandomizedSearch object
random_search = RandomizedSearchCV(
    model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42
)

# Fit it to the data and find the best hyperparameters
random_search.fit(x_train, y_train)

# Print the best parameters and the corresponding score
print("Best parameters: ", random_search.best_params_)
print("Best cross-validation score: ", random_search.best_score_)

# We can also use the best model found by RandomizedSearchCV
best_model = random_search.best_estimator_
