"""Merge scenarios and observations, normalize and flatten.

Created on Tue May  9 19:43:16 2023
@author: haduong@centre-cired.fr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from simulations import get_simulations
from observations import get_observations
from matplotlib.collections import LineCollection


# %% This is a list not a set, order matters when we store in an numpy array

indicators_simulations = [
    "Emissions|CO2",  # total net-CO2 emissions from all sources, Mt CO2/yr
    "GDP|MER",  # GDP at market exchange rate, billion US$2010/yr
    #    "GDP|PPP",
    "Population",
    "Primary Energy"  # ,
    #    "Secondary energy",
    #    "Final energy",
    #    "Capacity|Electricity",
    #    "Investment",
    #    "Consumption",
    #    "Land Cover|Cropland",
    #    "Land Cover|Pasture",
    #    "Land Cover|Forest",
]

indicators_observations = ["co2", "gdp", "population", "primary_energy_consumption"]

reference_levels = np.array(
    [  # World 1990 from ourworldindata.org
        27630,  # Emissions|CO2       Mt CO2/yr
        35850,  # GDP|MER             billion US$2010/yr
        5320,  # Population          million
        343.9,  # Primary Energy      EJ/yr,    95527 TWh
    ]
)

units_obs = pd.Series(reference_levels, index=indicators_observations)
units_sim = pd.Series(reference_levels, index=indicators_simulations)

# No need to normalize
# most of the data is in [0, 1] since we scale each variable
# compared the 'Value for World in 1990'.
# def _as_change(arrays):
#     """Convert a vector of levels into a vector of initial level and factors."""
#     rotated = np.roll(arrays, 1, axis=2)
#     rotated[:, :, 0] = 1
#     return arrays / rotated


def get_data(isim=None, iobs=None):
    """
    Combine simulations and observations.

    Label simulations as 0 and observations as 1.
    Return a tuple of two numpy arrays of the same length: data, labels.
    If called with no argument, will use all indicators.
    Assumes that the two arguments are aligned lists of variables.
    """
    if isim is None:
        isim = indicators_simulations
    if iobs is None:
        iobs = indicators_observations

    simulations = get_simulations(isim, units_sim)
    observations = get_observations(iobs, units_obs)

    print(len(simulations), "instances of simulations variables", isim)
    print(len(observations), "instances of observation variables", iobs)

    labels = np.concatenate([np.zeros(len(simulations)), np.ones(len(observations))])

    data = np.concatenate((simulations, observations))
    #   data = _as_change(data)
    data = data.reshape(data.shape[0], -1)

    return data, labels


# %% Verify the data. We expect to see normalization


def compare_data(sim="Population", obs="population"):
    data, labels = get_data([sim], [obs])

    num_obs = int(sum(labels))
    num_sim = int(len(labels) - num_obs)

    subsampling = 3
    data_sim = data[0:num_sim][::subsampling, :]
    data_obs = data[num_sim:][::subsampling, :]

    matrix1 = data_obs
    matrix2 = data_sim

    x = np.arange(matrix1.shape[1])

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # create 2 subplots side by side

    titles = ["Observations", "Simulations"]

    for idx, (ax, matrix) in enumerate(zip(axs, [matrix1, matrix2])):
        lines = [list(zip(x, matrix[i, :])) for i in range(matrix.shape[0])]

        lc = LineCollection(lines, linewidths=1, alpha=0.5)
        ax.add_collection(lc)

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(matrix.min(), matrix.max())

        # Set labels
        ax.set_xlabel("5 years period")
        ax.set_ylabel("Fraction of world 1990")
        ax.set_title(titles[idx])

    plt.suptitle(sim)
    plt.tight_layout()
    plt.show()


compare_data("Emissions|CO2", "co2")
compare_data("GDP|MER", "gdp")
compare_data("Population", "population")
compare_data("Primary Energy", "primary_energy_consumption")
