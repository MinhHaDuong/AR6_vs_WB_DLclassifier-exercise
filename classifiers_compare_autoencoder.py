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
from umap import UMAP

from classifier_others import model_dummy, model_xgb
from classifier_mlp import model_mlp

from classifiers_compare import compare, pretty_print

from log_config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


# Function for PCA dimensionality reduction
def perform_pca(x_raw_train, x_raw_test, n_components=15):
    logging.info(
        f"Performing PCA dimensionality reduction to {n_components} components."
    )
    pca = PCA(n_components=n_components)
    x_pca_train = pca.fit_transform(x_raw_train)
    x_pca_test = pca.transform(x_raw_test)
    return x_pca_train, x_pca_test


# Function for UMAP dimensionality reduction
def perform_umap(x_raw_train, x_raw_test, n_components=2, n_neighbors=15, min_dist=0.1):
    logging.info(
        f"Performing UMAP dimensionality reduction to {n_components} components."
    )
    umap = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
    x_umap_train = umap.fit_transform(x_raw_train)
    x_umap_test = umap.transform(x_raw_test)
    return x_umap_train, x_umap_test


# Function for autoencoder dimensionality reduction
def perform_autoencode(
    x_raw_train, x_raw_test, latent_dim=15, epochs=100, batch_size=32
):
    logging.info(f"Training an autoencoder to a {latent_dim} dimensions latent space.")
    # Create and train the autoencoder
    autoencoder = Sequential()
    autoencoder.add(Dense(16, activation="relu", input_shape=(x_raw_train.shape[1],)))
    autoencoder.add(Dense(latent_dim, activation="relu"))
    autoencoder.add(Dense(16, activation="relu"))
    autoencoder.add(Dense(x_raw_train.shape[1], activation="sigmoid"))
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

models_dict = {
    "Dummy baseline": model_dummy,
    "Gradient boosting machine": model_xgb,
    "Multilayer perceptron 256/0/128/0": model_mlp(
        x_raw_train.shape[1], 256, 0, 128, 0
    ),
    "bis": model_mlp(x_raw_train.shape[1], 256, 0, 128, 0),
    "ter": model_mlp(x_raw_train.shape[1], 256, 0, 128, 0),
}

results.loc["raw"] = compare(
    models_dict, x_raw_train, x_raw_test, y_train, y_test, parallelize=True
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_raw_train)
x_test_scaled = scaler.transform(x_raw_test)

smote = SMOTE(random_state=0)
x_train_scaled_resampled, y_train_resampled = smote.fit_resample(
    x_train_scaled, y_train
)

results.loc["normalized"] = compare(
    models_dict,
    x_train_scaled_resampled,
    x_test_scaled,
    y_train_resampled,
    y_test,
    parallelize=True,
)

x_pca_train, x_pca_test = perform_pca(x_train_scaled_resampled, x_test_scaled)
models_dict["Multilayer perceptron 256/0/128/0"] = model_mlp(
    x_pca_train.shape[1], 256, 0, 128, 0
)
models_dict["bis"] = model_mlp(x_pca_train.shape[1], 256, 0, 128, 0)
models_dict["ter"] = model_mlp(x_pca_train.shape[1], 256, 0, 128, 0)
results.loc["PCA"] = compare(
    models_dict, x_pca_train, x_pca_test, y_train_resampled, y_test, parallelize=False
)

x_umap_train, x_umap_test = perform_umap(x_train_scaled_resampled, x_test_scaled)
models_dict["Multilayer perceptron 256/0/128/0"] = model_mlp(
    x_umap_train.shape[1], 256, 0, 128, 0
)
models_dict["bis"] = model_mlp(x_umap_train.shape[1], 256, 0, 128, 0)
models_dict["ter"] = model_mlp(x_umap_train.shape[1], 256, 0, 128, 0)
results.loc["UMAP"] = compare(
    models_dict, x_umap_train, x_umap_test, y_train_resampled, y_test, parallelize=False
)

x_latent_train, x_latent_test = perform_autoencode(
    x_train_scaled_resampled, x_test_scaled
)
models_dict["Multilayer perceptron 256/0/128/0"] = model_mlp(
    x_latent_train.shape[1], 256, 0, 128, 0
)
models_dict["bis"] = model_mlp(x_latent_train.shape[1], 256, 0, 128, 0)
models_dict["ter"] = model_mlp(x_latent_train.shape[1], 256, 0, 128, 0)
results.loc["latent"] = compare(
    models_dict,
    x_latent_train,
    x_latent_test,
    y_train_resampled,
    y_test,
    parallelize=False,
)

# %% Print and save the results

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 10000)

keys = ["raw", "normalized", "PCA", "UMAP", "latent"]

print(pretty_print(results, keys, "to_string"))

with open("classifiers_compare_autoencoder.txt", "w", encoding="utf-8") as f:
    print(pretty_print(results, keys, "to_csv", sep="\t"), file=f)
