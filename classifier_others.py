"""
Provide pre-tuned classification models for our OWID vs. AR6 dataset.
Can also be tune the models manually, should the dataset change.
Warning: tuning hyperparameters is a slow affair.

Created on Thu May 18 16:08:01 2023

@author: haduong@centre-cired.fr

"""

from scipy.stats import randint, uniform

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import xgboost as xgb

from data import get_sets


def optimum(search):
    x_train, x_test, y_train, y_test = get_sets(
        diff=True, normalize=True, rebalance=True
    )
    search.fit(x_train, y_train)
    best_model = search.best_estimator_
    print(best_model)
    print("Best parameters: ", search.best_params_)
    print("Best cross-validation score: ", search.best_score_)
    return best_model


# %% Dummy classifier

model_dummy = DummyClassifier()


# %% Logistic regression classifier

model_lr = LogisticRegression(solver="liblinear", random_state=42, C=100, penalty="l2")


def model_lr_tuned():
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "penalty": ["l1", "l2"],
    }
    model = LogisticRegression(solver="liblinear", random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5)
    best_model = optimum(grid_search)
    return best_model


# %% Support vector machine classifier

model_svm = svm.SVC(
    kernel="rbf",
    random_state=42,
    C=100,
    gamma=1,
    class_weight="balanced",
)


def model_svm_tuned():
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": [1, 0.1, 0.01, 0.001],
        "kernel": ["linear", "rbf"],
    }
    model = svm.SVC(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5)
    best_model = optimum(grid_search)
    return best_model


# %% Random forest classifier

model_rf = RandomForestClassifier(
    bootstrap=False,
    max_depth=30,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=390,
    random_state=42,
)


def model_rf_tuned():
    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 4),
        "bootstrap": [True, False],
    }
    model = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(model, param_dist, cv=3, n_jobs=-1, verbose=2)
    best_model = optimum(random_search)
    return best_model


# %%  Gradient Boosting Machine classifier

model_xgb = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    eta=0.1,
    colsample_bytree=0.75,
    learning_rate=0.30,
    max_depth=5,
    n_estimators=171,
    subsample=0.84,
)


def model_xgb_tuned():
    param_dist = {
        "learning_rate": uniform(0.01, 0.3),
        "max_depth": randint(3, 10),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "n_estimators": randint(100, 500),
    }
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")
    random_search = RandomizedSearchCV(
        model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42
    )
    best_model = optimum(random_search)
    return best_model
