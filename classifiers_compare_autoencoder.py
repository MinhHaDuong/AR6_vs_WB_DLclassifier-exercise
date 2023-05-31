""" Autoencode IPAT sequences and see 

Created on Tue May 30 12:24:16 2023

@author: haduong@centre-cired.fr
"""

import logging
import pandas as pd

from data import get_sets, all_vars
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.decomposition import PCA

from classifier_others import model_dummy, model_xgb
from classifier_mlp import model_mlp

from classifiers_compare import compare, pretty_print

from log_config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

NEWDIM = 15


# Function for PCA dimensionality reduction
def perform_pca(x_raw_train, x_raw_test, n_components=NEWDIM):
    logging.info(
        f"Performing PCA dimensionality reduction to {n_components} components."
    )
    pca = PCA(n_components=n_components)
    x_pca_train = pca.fit_transform(x_raw_train)
    x_pca_test = pca.transform(x_raw_test)
    return x_pca_train, x_pca_test


# Function for autoencoder dimensionality reduction
def perform_autoencode(
    x_raw_train, x_raw_test, latent_dim=NEWDIM, epochs=50, batch_size=32, n=1
):
    logging.info(f"Training an autoencoder to a {latent_dim} dimensions latent space.")
    autoencoder = Sequential()
    autoencoder.add(Dense(20, activation="relu", input_shape=(x_raw_train.shape[1],)))
    if n == 2:
        autoencoder.add(Dense(20, activation="relu"))
    autoencoder.add(Dense(latent_dim, activation="linear"))
    autoencoder.add(Dense(20, activation="relu"))
    if n == 2:
        autoencoder.add(Dense(20, activation="relu"))
    autoencoder.add(Dense(x_raw_train.shape[1], activation="linear"))

    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(x_raw_train, x_raw_train, epochs=epochs, batch_size=batch_size)

    # Obtain the latent space representation
    encoder = Sequential(autoencoder.layers[:2])
    x_latent_train = encoder.predict(x_raw_train)
    x_latent_test = encoder.predict(x_raw_test)

    return x_latent_train, x_latent_test


# %% Run the comparison

results = pd.DataFrame(columns=["result", "duration"])

x_raw_train, x_raw_test, y_train, y_test = get_sets(all_vars, as_change=True)


def models_dict(input_dim=x_raw_train.shape[1]):
    return {
        "Dummy baseline": model_dummy,
        "Gradient boosting machine": model_xgb,
        "Multilayer perceptron": model_mlp(input_dim),
        "bis": model_mlp(input_dim),
        "ter": model_mlp(input_dim),
    }


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_raw_train)
x_test_scaled = scaler.transform(x_raw_test)
smote = SMOTE(random_state=0)
x_base_train, y_base_train = smote.fit_resample(x_train_scaled, y_train)

results.loc["normalized"] = compare(
    models_dict(), x_base_train, x_test_scaled, y_base_train, y_test
)

x_pca_train, x_pca_test = perform_pca(x_base_train, x_test_scaled)
assert x_pca_train.shape[1] == NEWDIM
results.loc["PCA"] = compare(
    models_dict(x_pca_train.shape[1]), x_pca_train, x_pca_test, y_base_train, y_test
)


x_latent_train, x_latent_test = perform_autoencode(x_base_train, x_test_scaled)
assert x_latent_train.shape[1] == NEWDIM
results.loc["latent"] = compare(
    models_dict(x_latent_train.shape[1]),
    x_latent_train,
    x_latent_test,
    y_base_train,
    y_test,
)

x_latent_train, x_latent_test = perform_autoencode(x_base_train, x_test_scaled, n=2)
assert x_latent_train.shape[1] == NEWDIM
results.loc["latent2"] = compare(
    models_dict(x_latent_train.shape[1]),
    x_latent_train,
    x_latent_test,
    y_base_train,
    y_test,
)

# %% Print and save the results

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 10000)

keys = ["normalized", "PCA", "latent", "latent2"]

print(pretty_print(results, keys, "to_string"))

with open("classifiers_compare_autoencoder.csv", "w", encoding="utf-8") as f:
    print(pretty_print(results, keys, "to_csv", sep="\t"), file=f)
