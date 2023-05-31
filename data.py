"""Merge scenarios and observations, normalize and flatten.

Created on Tue May  9 19:43:16 2023
@author: haduong@centre-cired.fr
"""

import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from owid_sequences import df_sequences as df_observations
from ar6_sequences import df_sequences as df_simulations

from log_config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


all_vars = ["co2", "gdp", "pop", "tpec"]


def dif(arrays):
    """Return the first difference of each array in the list.

    Drop rather than insert NaN, so returned arrays will be one column narrower."""
    rotated = np.roll(arrays, 1, axis=2)
    return (arrays - rotated)[:, :, 1:]


def flat(array):
    return array.reshape(array.shape[0], -1)


def sequence2array(df, indicators, group_keys):
    """Return data array after applying the select function."""

    assert indicators, "Indicators must be a non-empty list."
    assert not df.empty, "DataFrame to select from must not be empty."
    assert not df.isna().any().any(), "DataFrame must not contains NaN values"
    assert not (df == 0).any().any(), "DataFrame must not contains zero values"

    mask = pd.Series(
        df.index.get_level_values("variable").isin(indicators), index=df.index
    )
    group_counts = mask.groupby(level=group_keys).transform("sum")
    mask[group_counts != len(indicators)] = False

    subdf = df[mask].sort_index()

    assert set(subdf.index.get_level_values("variable").unique()) == set(
        indicators
    ), "Incorrect Variables selection."
    assert (
        not subdf.empty
    ), "The result is empty. Match the list of indicators to observations data."

    result = np.array([a for _, a in subdf.groupby(level=group_keys)])
    return result


def get_data(var=None, as_change=None, flatten=True):
    """
    Combine simulations and observations.

    Label simulations as 0 and observations as 1.
    Return a tuple of two numpy arrays of the same length: data, labels.
    If called with no argument, will use all indicators.
    """
    if var is None:
        var = all_vars

    if isinstance(var, pd.Index):
        var = var.tolist()

    observations = sequence2array(df_observations, var, ["country", "year"])

    simulations = sequence2array(
        df_simulations,
        var,
        ["Model", "Scenario", "countrycode", "year"],
    )

    logging.info(
        f"{len(observations)} observations \t{len(simulations)} simulations\tfor {var}"
    )

    labels = np.concatenate([np.zeros(len(simulations)), np.ones(len(observations))])

    data = np.concatenate((simulations, observations))

    if as_change:
        data = dif(data)

    if flatten:
        data = flat(data)

    return data, labels


def get_sets(var=None, as_change=None):
    data, labels = get_data(var, as_change)
    return train_test_split(data, labels, test_size=0.2, random_state=42)
