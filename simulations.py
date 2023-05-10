"""Use deep learning to distinguish AR6 scenarios from SDI observations.

Created on Thu Apr 28 2023

@author: haduong@centre-cired.fr
"""

import pandas as pd
import numpy as np

from ar6_scenario_database_iso3 import df, top_variables

# %% This is a list not a set, order matters when we store in an numpy array

indicators = [
    "Emissions|CO2",  # total net-CO2 emissions from all sources, Mt CO2/yr
    "GDP|MER",        # GDP at market exchange rate, billion US$2010/yr
    "Population",
    "Primary Energy"
    ]

"""
Note: VN Masterplan 2021-2030 objectives pertain to:
    Average GDP growth
    GDP per capita
    Urbanization rate
    Propostion of GDP in services / industry+construction / agriculture+forestry+fisheries
    Propostion of GDP of Digital economy
    Population size
    HDI
    Life Expectancy and Healthy life Expectancy
    Residential floor per person in urban area
    Hospital beds and doctors per 10.000 people
    Forest cover %, land reserves area, marine and coastal protected areas %
    Rate of wastewater treatment
    MSW collected %
    GHG intensity of GDP
"""

# Check we picked popular indicators
print('Number of scenario x country simulations available, by variable')
print(top_variables(-1)[indicators])
print()

# %% Keep only the rows describing the indicators


def hasall(df, indicators):
    """Return a mask so that the subdataframe df[mask] is 'full' wrt indicators.

    'Full' meaning for any triple (model, scenario, region),
    the set of variables V so that rows (model, scenario, region, V) are present
    is exactly the set 'indicators'.
    We return the largest such subdataframe.
    """
    mask = pd.Series(
        df.index.get_level_values('Variable').isin(indicators),
        index=df.index)
    group_counts = mask.groupby(level=[0, 1, 2]).transform('sum')
    mask[group_counts != len(indicators)] = False
    return mask

df = df[hasall(df, indicators)]


# %% Sanity check

units = df['Unit'].unique() 
print('Units:')
print(units)
print()
assert len(units) == len(indicators), "More units than variables."
del units


# %% Make a numpy array of 25-years trajectories

width = 6   # At five years step, twenty five years including extremities

simulations = np.array([
    a[:, i:i + width]
    for _, trajectory in df.groupby(level=[0, 1, 2])
    for a in [trajectory.iloc[:, 1:].values]
    for i in range(df.shape[1] - width)
    if not (np.isnan(a) | (a == 0)).any()
])

del width


# %%

try:
    np.save('simulations.npy', simulations)
    print('Array simulations saved successfully!')
except Exception as e:
    print('An error occurred while saving the array simulations:', e)