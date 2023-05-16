"""
Read the Our World In Data (OWID) CO2 dataset.
Select variables, change units, build 25-year trajectories with 5 year timestep
Expose as a Pandas DataFrame with multindex [country, variable, year]
and columns [value, value_Y+5, ..., value_Y+25]
Cache the result in a .pkl file

See OWID data pipeline https://docs.owid.io/projects/etl/en/latest/api/python/

Usage: from owid_trajectories import df

Created on Thu Apr 20 15:03:33 2023
@author: haduong@centre-cired.fr
"""


import pandas as pd

# Got it at  https://github.com/owid/co2-data/blob/master/owid-co2-data.csv
# Units at https://github.com/owid/co2-data/blob/master/owid-co2-codebook.csv
filename_raw = "owid-co2-data.csv"

filename_clean = "owid_trajectories.pkl"


def _get_dataframe(filename):
# Note: We would prefer to use nullable integers than floats but Pandas 1.x has only NaN floats.
    coltypes = {
        'country': 'category',
        'year': 'int',
        'population': 'float32',
        'gdp': 'float32',
        'co2': 'float32',
        'primary_energy_consumption': 'float32'}

    units = {
        'population': 1E6,                    # Population in Million
        'gdp': 1E9,                           # GDP in billion  $ (international 2011)
        'co2': 1,                             # CO2 Mt 
        'primary_energy_consumption': 277.8}  # Primary energy in Ej from TWh
    
    return pd.read_csv(
        filename_raw,
        index_col=[0, 1],
        usecols=coltypes.keys(),
        dtype=coltypes).div(pd.Series(units))

#%%

def _get_values_forward(group):
    group = group.reset_index()
    # Set 'Year' as the index so we can use .loc
    group = group.set_index('year')
    group['value_Y+5'] = group['value'].reindex(group.index + 5).values
    group['value_Y+10'] = group['value'].reindex(group.index + 10).values
    group['value_Y+15'] = group['value'].reindex(group.index + 15).values
    group['value_Y+20'] = group['value'].reindex(group.index + 20).values
    group['value_Y+25'] = group['value'].reindex(group.index + 25).values
    return group


def _shake(df):
    df_reset = df.reset_index()  # Reset the index so 'country' and 'year' become regular columns
    
    df_long = df_reset.melt(id_vars=['country', 'year'],
                            value_vars=df.columns,
                            var_name='variable',
                            value_name='value')

    # Now set 'country', 'year', and 'variable' as the new index
    df_long.set_index(['country', 'variable', 'year'], inplace=True)

    df_long = df_long.groupby(['country', 'variable']).apply(_get_values_forward)
    df_long = df_long.drop(df_long.columns[[0, 1]], axis=1).dropna()
    return df_long.reorder_levels(['country', 'year', 'variable']).sort_index()

# %%

try:
    df = pd.read_pickle(filename_clean)
    print('Successfully read OWID trajectories from file', filename_clean)
except:
    print('Unable to access ', filename_clean, '. Attempting to create it.')
    try:
        df = _shake(_get_dataframe(filename_raw))
        df.to_pickle(filename_clean)
        print('Cleaned OWID trajectories saved successfully!')
    except Exception as e:
        print('An error occurred while saving the OWID trajectories:', e)

        