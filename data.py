"""Merge scenarios and observations, normalize and flatten.

Created on Tue May  9 19:43:16 2023
@author: haduong@centre-cired.fr
"""

import pandas as pd
import numpy as np
from simulations import get_simulations
from observations import get_observations


# %% This is a list not a set, order matters when we store in an numpy array

indicators_simulations = [
    "Emissions|CO2",  # total net-CO2 emissions from all sources, Mt CO2/yr
    "GDP|MER",        # GDP at market exchange rate, billion US$2010/yr
#    "GDP|PPP",
    "Population",
    "Primary Energy" #,
#    "Secondary energy",
#    "Final energy",
#    "Capacity|Electricity",
#    "Investment",
#    "Consumption",
#    "Land Cover|Cropland",
#    "Land Cover|Pasture",
#    "Land Cover|Forest",
    ]

indicators_observations = [
    'co2',
    'gdp',
    'population',
    'primary_energy_consumption'
    ]

reference_levels = np.array([          # World 1990 from ourworldindata.org
    27630,  # Emissions|CO2       Mt CO2/yr
    35850,  # GDP|MER             billion US$2010/yr
    5320,   # Population          million
    343.9   # Primary Energy      EJ/yr,    95527 TWh
    ])

units_obs = pd.Series(reference_levels, index = indicators_observations)
units_sim = pd.Series(reference_levels, index = indicators_simulations)

# No need to normalize
# most of the data is in [0, 1] since we scale each variable
# compared the 'Value for World in 1990'.
# def _as_change(arrays):
#     """Convert a vector of levels into a vector of initial level and factors."""
#     rotated = np.roll(arrays, 1, axis=2)
#     rotated[:, :, 0] = 1
#     return arrays / rotated


def get_data(isim=indicators_simulations, iobs=indicators_observations):
    print('Simulations variables', isim)
    print('Observation variables', iobs)
    simulations = get_simulations(isim, units_sim)
    observations = get_observations(iobs, units_obs)

    labels = np.concatenate([
        np.zeros(len(simulations)),
        np.ones(len(observations))])

    data = np.concatenate((simulations, observations))
#   data = _as_change(data)
    data = data.reshape(data.shape[0], -1)

    return data, labels
