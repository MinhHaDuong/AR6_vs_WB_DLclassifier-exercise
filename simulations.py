"""Use deep learning to distinguish AR6 scenarios from SDI observations.

Created on Thu Apr 28 2023

@author: haduong@centre-cired.fr
"""

import pandas as pd
import numpy as np
import time

from ar6_scenario_database_iso3 import df, top_variables

#print(df2.info())

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

df2 = df[hasall(df, indicators)]
#print(df2.info())


# %% Make a numpy array with a sliding window

gen_length = 6   # At five years step, twenty five years including extremities
num_cols = df2.shape[1]
multiindex = df2.index.droplevel('Variable').unique()

reference_units = df2.loc[multiindex[0]]["Unit"]
print("Reference units:", reference_units)
print()


 
print('Starting long loop on scenarios')
start = time.time()

simulations = []

for i in multiindex:
    trajectory = df2.loc[i]
    assert list(trajectory.index.values) == indicators
    assert (trajectory['Unit'] == reference_units).all()
    for y in range(1, num_cols - gen_length + 1):
        a = trajectory.iloc[:, y:y + gen_length].values
        if np.all(a != 0) and not np.any(np.isnan(a)):
             simulations.append(a)

del a, i, y, multiindex, trajectory, gen_length, num_cols, reference_units

simulations = np.array(simulations)

assert not np.isnan(simulations).any(), "Array contains null values"
assert (simulations != 0).all(), "Array containts zero values"

end = time.time()
print("Done. Execution time: ", end - start)
print()
del start, end

expected = simulations

# %% Make a numpy array with a sliding window FASTER

gen_length = 6   # At five years step, twenty five years including extremities
num_cols = df2.shape[1]
multiindex = df2.index.droplevel('Variable').unique()

reference_units = df2.loc[multiindex[0]]["Unit"]
# print("Reference units:", reference_units)
# print()


 
print('Starting long loop on scenarios')
start = time.time()

simulations = []

for i, trajectory in df2.groupby(level=[0, 1, 2]):
    trajectory = trajectory.reset_index(level=[0, 1, 2], drop=True)
    assert list(trajectory.index.get_level_values('Variable')) == indicators
    assert (trajectory['Unit'] == reference_units).all()
    for y in range(1, num_cols - gen_length + 1):
        a = trajectory.iloc[:, y:y + gen_length].values
        if np.all(a != 0) and not np.any(np.isnan(a)):
             simulations.append(a)

if (np.array(simulations) == expected).all():
    print('No regression')
else:
    print('Failed')

end = time.time()
print("Execution time: ", end - start)
print()
del start, end


# %%

try:
    np.save('simulations.npy', simulations)
    print('Array simulations saved successfully!')
except Exception as e:
    print('An error occurred while saving the array simulations:', e)