""" Separate AR6 scenarios from past observations

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
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

from data import get_data, indicators_simulations, indicators_observations


# Multilayers perceptron


def classify(isim=None, iobs=None):
    if isim is None:
        isim = indicators_simulations
    if iobs is None:
        iobs = indicators_observations

    data, labels = get_data(isim, iobs)

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    model = Sequential()
    model.add(Dense(96, activation="relu", input_dim=data.shape[1]))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.01),
        metrics=[
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc"),
            tfa.metrics.F1Score(
                name="f1_score", num_classes=1, threshold=0.5
            ),  # Adding F1 score
        ],
    )

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

    # Use the model to make predictions on the test set
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
    return score, model


# Exemples

classify()

classify(["Population"], ["population"])

# %% Tune the model on the complete variables case

data, labels = get_data()

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)


def build_model(hp):
    model = tf.keras.models.Sequential()
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


tuner = RandomSearch(
    build_model,
    objective=Objective("val_auc", direction="max"),
    max_trials=50,
    directory="my_dir",
    project_name="tune_model",
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

tuner.results_summary()

"""

Trial 47 summary
Hyperparameters:
units_1: 96
dropout_1: 0.0
units_2: 128
dropout_2: 0.30000000000000004
learning_rate: 0.01
Score: 0.9995813965797424

Trial 35 summary
Hyperparameters:
units_1: 192
dropout_1: 0.0
units_2: 96
dropout_2: 0.2
learning_rate: 0.01
Score: 0.9989245533943176

Trial 41 summary
Hyperparameters:
units_1: 192
dropout_1: 0.2
units_2: 96
dropout_2: 0.30000000000000004
learning_rate: 0.01
Score: 0.9988268613815308

Trial 25 summary
Hyperparameters:
units_1: 192
dropout_1: 0.0
units_2: 128
dropout_2: 0.30000000000000004
learning_rate: 0.001
Score: 0.9987591505050659

Trial 12 summary
Hyperparameters:
units_1: 192
dropout_1: 0.1
units_2: 96
dropout_2: 0.4
learning_rate: 0.01
Score: 0.9987215995788574

Trial 17 summary
Hyperparameters:
units_1: 224
dropout_1: 0.2
units_2: 64
dropout_2: 0.1
learning_rate: 0.01
Score: 0.9985978603363037

Trial 20 summary
Hyperparameters:
units_1: 192
dropout_1: 0.30000000000000004
units_2: 96
dropout_2: 0.0
learning_rate: 0.001
Score: 0.9985247254371643

Trial 37 summary
Hyperparameters:
units_1: 128
dropout_1: 0.1
units_2: 64
dropout_2: 0.4
learning_rate: 0.001
Score: 0.9983959197998047

Trial 22 summary
Hyperparameters:
units_1: 128
dropout_1: 0.2
units_2: 96
dropout_2: 0.2
learning_rate: 0.001
Score: 0.9982591271400452

Trial 39 summary
Hyperparameters:
units_1: 32
dropout_1: 0.1
units_2: 96
dropout_2: 0.0
learning_rate: 0.01
Score: 0.998193621635437
"""
