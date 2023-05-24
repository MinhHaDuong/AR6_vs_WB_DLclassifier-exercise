"""
Read the Our World In Data (OWID) CO2 dataset.
Select variables, change units, build 25-year sequences with 5 year timestep
Expose as a Pandas DataFrame with multindex [country, variable, year]
and columns [value, value_Y+5, ..., value_Y+25]
Cache the result in a .pkl file

See OWID data pipeline https://docs.owid.io/projects/etl/en/latest/api/python/

Usage: from owid_sequences import df_sequences as df_observations

Created on Thu Apr 20 15:03:33 2023
@author: haduong@centre-cired.fr
"""

import pickle
import pandas as pd

# Got it at  https://github.com/owid/co2-data/blob/master/owid-co2-data.csv
# Units at https://github.com/owid/co2-data/blob/master/owid-co2-codebook.csv
FILENAME_RAW = "owid-co2-data.csv"
FILENAME_CLEAN = "owid_sequences.pkl"
FILENAME_CUTYEARS = "owid_cutyears.csv"
FILENAME_NOTCOUNTRY = "owid_notcountry.csv"


def get_dataframe(filename, censored_countrynames=[], cut_years={}):
    # Note: We would prefer to use nullable integers than floats but Pandas 1.x has only NaN floats.
    coltypes = {
        "country": "category",
        "year": "int",
        "population": "float32",
        "gdp": "float32",
        "co2": "float32",
        "primary_energy_consumption": "float32",
    }

    df = pd.read_csv(
        filename, index_col=[0, 1], usecols=coltypes.keys(), dtype=coltypes
    )

    df = df.rename(
        columns={
            "population": "pop",
            "gdp": "gdp",
            "co2": "co2",
            "primary_energy_consumption": "tpec",
        }
    )

    units = {
        "pop": 1e6,  # Population in Million
        "gdp": 1e9,  # GDP in billion  $ (international 2011)
        "co2": 1,  # CO2 Mt
        "tpec": 277.8,
    }  # Primary energy in Ej from TWh

    df = df.div(pd.Series(units))

    mask = ~df.index.get_level_values("country").isin(censored_countrynames)

    for country, cut_year in cut_years.items():
        country_mask = df.index.get_level_values("country") == country
        year_mask = df.index.get_level_values("year") <= cut_year
        censored = country_mask & year_mask
        mask = mask & (~censored)

    df = df[mask]
    return df


# %%


def _get_values_forward(group):
    group = group.reset_index().set_index("year")
    group = group.drop(columns=["country", "variable"])
    group["value_Y+5"] = group["value"].reindex(group.index + 5).values
    group["value_Y+10"] = group["value"].reindex(group.index + 10).values
    group["value_Y+15"] = group["value"].reindex(group.index + 15).values
    group["value_Y+20"] = group["value"].reindex(group.index + 20).values
    group["value_Y+25"] = group["value"].reindex(group.index + 25).values
    return group


def shake(df):
    # Melt the dataframe from wide to long format
    result = df.reset_index().melt(
        id_vars=["country", "year"],
        value_vars=df.columns,
        var_name="variable",
        value_name="value",
    )
    result.set_index(["country", "variable", "year"], inplace=True)

    # Create the trajectories
    result = result.groupby(["country", "variable"]).apply(_get_values_forward)
    result = result.reorder_levels(["country", "year", "variable"]).sort_index()

    # Cleanup trajectories with NaNs and zeros
    result = result.dropna()
    result = result[(result != 0).all(axis=1)]

    return result


# %%

try:
    df_sequences = pd.read_pickle(FILENAME_CLEAN)
    print("Successfully read OWID sequences from file", FILENAME_CLEAN)
except (IOError, EOFError, pickle.UnpicklingError) as e_read:
    print(
        "Unable to access ", FILENAME_CLEAN, ":", e_read, ".\nAttempting to create it."
    )
    try:
        with open(FILENAME_NOTCOUNTRY, "r") as file:
            not_country = [line.strip() for line in file]
        cut_years = pd.read_csv(FILENAME_CUTYEARS, index_col=0)
        cut_years = cut_years[cut_years.columns[0]]
        df_filtered = get_dataframe(FILENAME_RAW, not_country, cut_years)
        df_sequences = shake(df_filtered)
        df_sequences.to_pickle(FILENAME_CLEAN)
        print("Cleaned OWID sequences saved successfully!")
    except Exception as e:
        print("An error occurred while saving the OWID trajectories:", e)