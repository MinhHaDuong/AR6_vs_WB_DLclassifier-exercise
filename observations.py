"""
Access the Our World In Data (OWID) as a dataframe.

See OWID data pipeline https://docs.owid.io/projects/etl/en/latest/api/python/

Created on Thu Apr 20 15:03:33 2023
@author: haduong
"""


import pandas as pd
import numpy as np

# Got it at  https://github.com/owid/co2-data/blob/master/owid-co2-data.csv
# Units at https://github.com/owid/co2-data/blob/master/owid-co2-codebook.csv
filename = "owid-co2-data.csv"

coltypes = {
    'country': 'category',
    'year': 'int',
    'population': 'float32',   # Known gotcha: NaN is a float
    'gdp': 'float32',
    'co2': 'float32',
    'primary_energy_consumption': 'float32'}

df = pd.read_csv(
    filename,
    index_col=[0, 1],
    usecols=coltypes.keys(),
    dtype=coltypes)

del filename, coltypes

df = df.dropna()

units = [
    1E6,    # Population in Million
    1E9,    # GDP in billion  $ (international 2011)
    1,      # CO2 Mt 
    277.8]  # Primary energy in Ej from TWh

df = df.div(units, axis=1)

del units

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', None)

# print(df)

# %% Build the array with the country development trajectories

def trajectories(country, min_years=25):
    block = df.loc[country]
    years = block.index
    start, end = years[0], years[-1]
    if end - start < min_years:
        return []
    result = []
    for y in range(start, end - min_years + 1):
        grid = list(range(y, y + min_years + 1, 5))
        try:
            a = block.loc[grid].values.transpose()
            if np.all(a != 0) and not np.any(np.isnan(a)):
                result.append(a)
        except KeyError:
            pass
    return result

# trajectories('Afghanistan')
# trajectories('World')

observations = []

for country in df.index.get_level_values('country'):
    observations.extend(trajectories(country))

del country

observations = np.array(observations, dtype='float32')

# %% Save te result

assert not np.isnan(observations).any(), "Array contains null values"
assert (observations != 0).all(), "Array containts zero values"

try:
    np.save('observations.npy', observations)
    print('Array observations saved successfully!')
except Exception as e:
    print('An error occurred while saving the array observations:', e)