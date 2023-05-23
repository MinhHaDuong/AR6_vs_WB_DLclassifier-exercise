"""Merge scenarios and observations, normalize and flatten.

Created on Tue May  9 19:43:16 2023
@author: haduong@centre-cired.fr
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from simulations import get_simulations
from observations import get_observations


# Source ourworldindata.org
world_1990 = pd.Series({
        "co2": 27630,   # Mt CO2/yr
        "gdp": 35850,   # billion US$2010/yr
        "pop": 5320,    # million people
        "tpec": 343.9,  # EJ/yr, 95527 TWh
    })

all_vars = world_1990.index

def _as_change(arrays):
    """Convert a vector of levels into a vector of initial level and factors."""
    rotated = np.roll(arrays, 1, axis=2)
    #    rotated[:, :, 0] = 1
    return (arrays / rotated)[:, :, 1:]


def get_data(var=None, as_change=None):
    """
    Combine simulations and observations.

    Label simulations as 0 and observations as 1.
    Return a tuple of two numpy arrays of the same length: data, labels.
    If called with no argument, will use all indicators.
    Assumes that the two arguments are aligned lists of variables.
    """
    if var is None:
        var = all_vars

    simulations = get_simulations(var, units=world_1990)
    observations = get_observations(var, units=world_1990)

    print(f"{var}    {len(simulations)} simulations, {len(observations)} observation sequences")

    labels = np.concatenate([np.zeros(len(simulations)), np.ones(len(observations))])

    data = np.concatenate((simulations, observations))
    if as_change:
        data = _as_change(data)
    data = data.reshape(data.shape[0], -1)

    return data, labels


def get_sets():
    data, labels = get_data()
    return train_test_split(data, labels, test_size=0.2, random_state=42)
