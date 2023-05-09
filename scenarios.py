#!/home/haduong/CNRS/papiers/actif/DL_SDI/src/classifying_country_scenarios/venv/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 2023

@author: haduong

Based on tutorial MNIST1 from FIDLE

This file is an exercise to manipulate IPCC AR6 scenarios database
"""

import pandas as pd
import numpy as np
import time

from ar6_scenario_database_iso3 import df, top_variables

# %%

print(df.info())

# %% This is a list, not a set because later we store in an numpy array
# so order must be preserved.

indicators = [
    "Emissions|CO2",
    "GDP|MER",
    #    "Land Cover|Forest",
    "Population",
    #    "Population|Urban",
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
print(top_variables(-1)[indicators])


# %% Keep only the rows describing the indicators


def hasall(indicators):
    """Return a mask so that the subdataframe df[mask] is 'full' wrt indicators.

    'Full' meaning for any triple (model, scenario, region),
    the set of variables V so that rows (model, scenario, region, V) are present
    is exactly the set 'indicators'.
    We return the largest such subdataframe.
    """
    mask = pd.Series(False, index=df.index)
    for (m, s, r) in df.index.droplevel('Variable').unique():
        mi = pd.MultiIndex.from_product([[m], [s], [r], indicators])
        if mi.isin(df.index).all():
            mask[mi] = True
    return mask


start = time.time()
df2 = df[hasall(indicators)]
end = time.time()

print("Execution time: ", end - start)
del start, end
print(df2.info())


# %%

gen_length = 6   # At five years step, twenty five years including extremities
num_cols = df2.shape[1]
multiindex = df2.index.droplevel('Variable').unique()

reference_units = df2.loc[multiindex[0]]["Unit"]
print("Reference units:", reference_units)

simulations = []

for i in multiindex:
    trajectory = df2.loc[i]
    assert list(trajectory.index.values) == indicators
    assert (trajectory['Unit'] == reference_units).all()
    for y in range(1, num_cols - gen_length + 1):
        a = trajectory.iloc[:, y:y + gen_length].values
        if not np.isnan(a).any():           # Todo: clean the NaN and the zeros afterwards
            simulations.append(a)

del a, i, y, multiindex, trajectory, gen_length, num_cols, reference_units

simulations = np.array(simulations)

assert not np.isnan(simulations).any(), "Array contains null values"
assert (simulations != 0).all(), "Array containts zero values"

# %% Normalization
"""The series in levels is transformed to a series where:
   - The first value is a fraction of the world in 1990
   - Subsequent values are factors from the previous value
   so that is A is the original and B the transformed vector
     a_i is the product from b_0  to  b_i

   This ensures that the values are around 1.
"""


def as_change(a):
    """Convert a vector of levels into a vector of initial level and factors."""
    b = np.zeros_like(a)
    b[0] = a[0]
    for i in range(1, len(a)):
        if abs(a[i - 1]) <= 0.001:
            print("Warning: division by ", a[i-1], " at i=", i)
        b[i] = a[i] / a[i - 1]
    return b


normalized = np.apply_along_axis(as_change, 2, simulations)

world1990 = np.array([          # Source: ourworldindata.org
    27630,  # Emissions|CO2       Mt CO2/yr
    35850,  # GDP|MER             billion US$2010/yr
    5320,   # Population          million
    343.9])   # Primary Energy      EJ/yr,    95527 TWh

normalized[:, :, 0] /= world1990
