"""Merge scenarios and observations, normalize and flatten.

The series in levels is transformed to a series where:
   - The first value is a fraction of the world in 1990
   - Subsequent values are factors from the previous value
   so that is A is the original and B the transformed vector
     a_i is the product from b_0  to  b_i

   This ensures that the values are around 1.

Created on Tue May  9 19:43:16 2023
@author: haduong@centre-cired.fr
"""

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

 #TODO: Use a dict and select the proper subset in get_data /=
reference_levels = np.array([          # World 1990 from ourworldindata.org
    27630,  # Emissions|CO2       Mt CO2/yr
    35850,  # GDP|MER             billion US$2010/yr
    5320,   # Population          million
    343.9   # Primary Energy      EJ/yr,    95527 TWh
    ])


def _as_change(arrays):
    """Convert a vector of levels into a vector of initial level and factors."""
    rotated = np.roll(arrays, 1, axis=2)
    rotated[:, :, 0] = 1
    return arrays / rotated


def get_data(isim=indicators_simulations, iobs=indicators_observations):
    print('Simulations variables', isim)
    print('Observation variables', iobs)
    simulations = get_simulations(isim)
    observations = get_observations(iobs)

    labels = np.concatenate([
        np.zeros(len(simulations)),
        np.ones(len(observations))])

    data = np.concatenate((simulations, observations))
    data = _as_change(data)
    data[:, :, 0] /= reference_levels
    data = data.reshape(data.shape[0], -1)

    return data, labels
