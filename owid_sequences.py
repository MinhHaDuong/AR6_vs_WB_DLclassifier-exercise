"""
Read the Our World In Data (OWID) CO2 dataset.
Select variables, change units, build 25-year sequences with 5 year timestep
Expose as a Pandas DataFrame with multindex [country, variable, year]
and columns [value, value_Y+5, ..., value_Y+25]
Cache the result in a .pkl file

See OWID data pipeline https://docs.owid.io/projects/etl/en/latest/api/python/

Usage:
Run standalone to create owid_sequences.pkl if it does not exist yet.
Import as a module   from owid_sequences import get_sequences
(exposes a getter not the df_sequence for lazy filecheck).

Created on Thu Apr 20 15:03:33 2023
@author: haduong@centre-cired.fr
"""

import logging
import pandas as pd
from functools import lru_cache

from log_config import setup_logger
from utils import cache

setup_logger()
logger = logging.getLogger(__name__)

# Got it at  https://github.com/owid/co2-data/blob/master/owid-co2-data.csv
# Units at https://github.com/owid/co2-data/blob/master/owid-co2-codebook.csv
FILENAME_IN = "data/owid-co2-data.csv"
FILENAME_NOTCOUNTRY = "data/owid_notcountry.csv"


def get_dataframe(filename, censored_countrynames=[]):
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

    df = df[mask]
    return df


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


@cache(__file__)
@lru_cache(maxsize=128)
def get_sequences():
    with open(FILENAME_NOTCOUNTRY, "r") as file:
        not_country = [line.strip() for line in file]
    df_filtered = get_dataframe(FILENAME_IN, not_country)
    df_sequences = shake(df_filtered)
    return df_sequences


# When run directly, create the .pkl if necessary
if __name__ == "__main__":
    get_sequences()
