""" Define a multilayer perceptron classifier to separate AR6 scenarios from past observations

Created on Tue May  9 21:02:33 2023
@author: haduong@centre-cired.fr
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow.keras import layers, metrics
import tensorflow_addons as tfa
from keras_tuner import RandomSearch
from keras_tuner.engine.objective import Objective
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

from data import get_data, get_sets


# Multilayers perceptron


def model_mlp(input_dim):
    model = Sequential()
    model.add(Dense(224, activation="relu", input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.01),
        metrics=[
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc"),
            tfa.metrics.F1Score(name="f1_score", num_classes=1, threshold=0.5),
        ],
    )

    return model


# %% Example


def train_model(model, x_train, y_train, x_test, y_test):
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, verbose=1, restore_best_weights=True
    )

    model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=30,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping],
    )

    return model


def test_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred = [round(pred[0]) for pred in y_pred]

    # Print a classification report
    print(classification_report(y_test, y_pred))

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print("Test precision:", score[2])
    print("Test recall:", score[3])
    print("Test AUC:", score[4])
    print("Test F1 Score:", score[5])
    return score


def define_train_test():
    x_train, x_test, y_train, y_test = get_sets()
    model = model_mlp(x_train.shape[1])
    model = train_model(model, x_train, y_train, x_test, y_test)
    score = test_model(model, x_test, y_test)

    return score, model


# define_train_test()


# %% Tune the model on the complete variables case


def model_mlp_tunable(hp):
    model = tf.keras.models.Sequential()

    data, labels = get_data()
    model.add(
        layers.Dense(
            units=hp.Int("units_1", min_value=32, max_value=256, step=32),
            activation="relu",
            input_dim=data.shape[1],
        )
    )
    model.add(
        layers.Dropout(hp.Float("dropout_1", min_value=0.0, max_value=0.5, step=0.1))
    )
    model.add(
        layers.Dense(
            units=hp.Int("units_2", min_value=32, max_value=128, step=32),
            activation="relu",
        )
    )
    model.add(
        layers.Dropout(hp.Float("dropout_2", min_value=0.0, max_value=0.5, step=0.1))
    )
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
        metrics=[
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc"),
            tfa.metrics.F1Score(name="f1_score", num_classes=1, threshold=0.5),
        ],
    )

    return model


def tune_mlp():
    x_train, x_test, y_train, y_test = get_sets()

    tuner = RandomSearch(
        model_mlp_tunable,
        objective=Objective("val_auc", direction="max"),
        max_trials=50,
        directory="my_dir",
        project_name="MLP_tuning",
    )

    tuner.search_space_summary()

    tuner.search(
        x_train,
        y_train,
        epochs=20,
        validation_data=(x_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, verbose=1, restore_best_weights=True
            )
        ],
    )

    print(tuner.results_summary())
    return tuner.results_summary()
