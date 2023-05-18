""" Separate AR6 scenarios from past observations

Created on Tue May  9 21:02:33 2023
@author: haduong@centre-cired.fr
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from data import get_data, indicators_simulations, indicators_observations


def classify(isim=indicators_simulations, iobs=indicators_observations):
    data, labels = get_data(isim, iobs)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    model = Sequential()
    model.add(Dense(128, activation="relu", input_dim=data.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, verbose=1, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=20,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
    )

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return score


# Exemples

# classify()

# classify(['Population'], ['population'])
