"""Compare methods to reduce data dimensionality

Created on Tue May 30 12:24:16 2023

@author: haduong@centre-cired.fr
"""

import logging
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.decomposition import PCA

from data import get_sets, all_vars

from classifier_others import model_dummy, model_xgb
from classifier_mlp import model_mlp

from compare import compare

from utils import cache
from log_config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

NEWDIM = 15


# Function for PCA dimensionality reduction
def perform_pca(x_raw_train, x_raw_test):
    logging.info("Performing PCA dimensionality reduction to %d components.", NEWDIM)
    pca = PCA(n_components=NEWDIM)
    x_pca_train = pca.fit_transform(x_raw_train)
    x_pca_test = pca.transform(x_raw_test)
    return x_pca_train, x_pca_test


# Function for autoencoder dimensionality reduction
def perform_autoencode(x_raw_train, x_raw_test, layers=1):
    logging.info(
        "Training autoencoder: dim %d -> dim %d.", x_raw_train.shape[1], NEWDIM
    )
    autoencoder = Sequential()
    autoencoder.add(Dense(20, activation="relu", input_shape=(x_raw_train.shape[1],)))
    autoencoder.add(Dropout(0.05))
    if layers == 2:
        autoencoder.add(Dense(20, activation="relu"))
        autoencoder.add(Dropout(0.05))
    autoencoder.add(Dense(NEWDIM, activation="linear"))
    autoencoder.add(Dense(20, activation="relu"))
    if layers == 2:
        autoencoder.add(Dense(20, activation="relu"))
    autoencoder.add(Dense(x_raw_train.shape[1], activation="linear"))

    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(
        x_raw_train,
        x_raw_train,
        epochs=100,
        batch_size=32,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
        validation_data=(x_raw_test, x_raw_test),
    )

    # Obtain the latent space representation
    encoder = Sequential(autoencoder.layers[: (1 + 2 * layers)])
    x_latent_train = encoder.predict(x_raw_train)
    x_latent_test = encoder.predict(x_raw_test)

    return x_latent_train, x_latent_test


@cache(__file__)
def get_results():
    results = pd.DataFrame(columns=["result", "duration"])

    x_base_train, x_base_test, y_train, y_test = get_sets(
        all_vars, diff=True, normalize=True, rebalance=True
    )

    def models_dict(input_dim=x_base_train.shape[1]):
        return {
            "Dummy baseline": model_dummy,
            "Gradient boosting machine": model_xgb,
            "Multilayer perceptron": model_mlp(input_dim),
            "bis": model_mlp(input_dim),
            "ter": model_mlp(input_dim),
        }

    results.loc["normalized"] = compare(
        models_dict(), (x_base_train, x_base_test, y_train, y_test)
    )

    x_pca_train, x_pca_test = perform_pca(x_base_train, x_base_test)
    assert x_pca_train.shape[1] == NEWDIM
    results.loc["PCA"] = compare(
        models_dict(NEWDIM), (x_pca_train, x_pca_test, y_train, y_test)
    )

    x_latent_train, x_latent_test = perform_autoencode(x_base_train, x_base_test)
    assert x_latent_train.shape[1] == NEWDIM
    results.loc["latent"] = compare(
        models_dict(NEWDIM), (x_latent_train, x_latent_test, y_train, y_test)
    )

    x_latent2_train, x_latent2_test = perform_autoencode(
        x_base_train, x_base_test, layers=2
    )

    assert x_latent2_train.shape[1] == NEWDIM
    results.loc["latent2"] = compare(
        models_dict(NEWDIM), (x_latent2_train, x_latent2_test, y_train, y_test)
    )

    return results


# When run directly, create the .pkl if necessary
if __name__ == "__main__":
    get_results()
