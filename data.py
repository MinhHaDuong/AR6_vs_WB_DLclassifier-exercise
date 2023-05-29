"""Merge scenarios and observations, normalize and flatten.

Created on Tue May  9 19:43:16 2023
@author: haduong@centre-cired.fr
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from owid_sequences import df_sequences as df_observations
from ar6_sequences import df_sequences as df_simulations


# Source ourworldindata.org
world_1990 = pd.Series(
    {
        "co2": 27630,  # Mt CO2/yr
        "gdp": 35850,  # billion US$2010/yr
        "pop": 5320,  # million people
        "tpec": 343.9,  # EJ/yr, 95527 TWh
    }
)

all_vars = world_1990.index


def dif(arrays):
    """Return the first difference of each array in the list.

    Drop rather than insert NaN, so returned arrays will be one column narrower."""
    rotated = np.roll(arrays, 1, axis=2)
    return (arrays - rotated)[:, :, 1:]


def flat(array):
    return array.reshape(array.shape[0], -1)


def sequence2array(df, indicators, units, group_keys):
    """Return data array after applying the select function and adjusting units."""

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

    subdf = subdf.div(units, level="variable", axis=0)

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

    observations = sequence2array(df_observations, var, world_1990, ["country", "year"])

    simulations = sequence2array(
        df_simulations,
        var,
        world_1990,
        ["Model", "Scenario", "countrycode", "year"],
    )

    print(
        f"{var}\tobservations:\t{len(observations)}\tsimulations:\t{len(simulations)}"
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
