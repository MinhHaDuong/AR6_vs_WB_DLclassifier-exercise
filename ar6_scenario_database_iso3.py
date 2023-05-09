"""
Access IPCC AR6 scenarios database of national results as a dataframe.

Expose scenario results as a Pandas dataframe.
The multiindex is Model - Scenario - Region - Variable.
Clean data to ensure consistent units.
Keep years at a five time step between 2005 and 2050.
Provide auxilliary functions to explore the list of variables.
Database online at https://data.ece.iiasa.ac.at/ar6/#/docs.

Example use:
from ar6_scenario_database_iso3 import df

Created on Thu Apr 28 2023
@author: haduong@centre-cired.fr
"""

import pandas as pd
from functools import lru_cache

filename = "AR6_Scenarios_Database_ISO3_v1.1.csv"

# %% Import the data


@lru_cache(maxsize=None)
def _get_dataframe():
    coltypes = {
        'Model': 'category',
        'Scenario': 'category',
        'Region': 'category',
        'Variable': 'category',
        'Unit': 'category',
        '2005': 'float32',
        '2010': 'float32',
        '2015': 'float32',
        '2020': 'float32',
        '2025': 'float32',
        '2030': 'float32',
        '2035': 'float32',
        '2040': 'float32',
        '2045': 'float32',
        '2050': 'float32'}

    return pd.read_csv(
        filename,
        index_col=[0, 1, 2, 3],
        usecols=[0, 1, 2, 3, 4, 8, 11, 16, 18, 23, 24, 25, 26, 27, 28],
        dtype=coltypes)


def _check_units(data):
    """Verify if units were used consistently in the table.

    Return 0 if okay, otherwise list variables expressed in more than one unit,
    and return their number.
    """
    s = data['Unit'].droplevel([0, 1, 2]).groupby(level=0).unique()

    offending_rows = s[s.apply(lambda x: len(x) > 1)]

    if len(offending_rows):
        print(offending_rows)
    return len(offending_rows)


def _clean(df):
    """Fix units in IPCC AR6 scenarios database.

    Unfixed: MESSAGEix-GLOBIOM_1.0 report 0 GDP|MER in 2005 in lieu of nan
    """
    df['Unit'] = df['Unit'].str.replace('Million', 'million')
    df['Unit'] = df['Unit'].str.replace('Int$2010', 'US$2010')
    df['Unit'] = df['Unit'].str.replace('million full-time equivalent', 'million')
    df.loc[df['Unit'] == 'PJ/yr', "2005":"2050"] *= 0.001
    df['Unit'] = df['Unit'].str.replace('PJ/yr', 'EJ/yr')
    df.loc[df['Unit'] == 'million tkm', "2005":"2050"] *= 0.001
    df['Unit'] = df['Unit'].str.replace('million tkm', 'bn tkm/yr')
    df.loc[df['Unit'] == 'million pkm', "2005":"2050"] *= 0.001
    df['Unit'] = df['Unit'].str.replace('million pkm', 'bn pkm/yr')
    df = df.drop(df[df.index.get_level_values('Variable') == 'Price|Carbon'].index)
    if _check_units(df):
        print("Alert: At least one Variable using more than one Unit.")
    return df


df = _clean(_get_dataframe())

models = df.index.get_level_values(0).unique().tolist()
scenarios = df.index.get_level_values(1).unique().tolist()
regions = df.index.get_level_values(2).unique().tolist()
variables = df.index.get_level_values(3).unique().tolist()
years = df.columns.tolist()[1:]


# %% Tool functions to explore the variables


def filter_variables(substring):
    """Filter variable names containing a substring.

    Useful to find the indicator we want in the large number of variables.
    """
    return list(filter(lambda s: substring in s, variables))


def top_variables(n):
    """Count the most frequently reported variables."""
    count = df.groupby(level=3).size()
    return count[variables].sort_values(ascending=False)[0:n]


def root_variables():
    """List the variables at the root of the variables nomenclature and counts.

    This shows that some root variables are rarely reported.
    """
    roots = [s for s in variables if '|' not in s]
    return top_variables(-1)[roots].sort_values(ascending=False)


# %% Functional access API mimicking module  world-bank-data

# def get_models():
#     return models

# def get_regions():
#     return regions

# def get_scenarios():
#     return scenarios

# def get_years():
#     return years
