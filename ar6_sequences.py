""" Align the simulations dataframe: return a dataframe of 25-year sequences
Drop uninteresting variables
Cache result on disk because the input df_trajectories has 3.9 million rows.

Created on Tue May 23 12:57:57 2023

@author: haduong@centre-cired.fr
"""

import logging
from functools import lru_cache
import multiprocessing as mp  # Breaks Spyder profilers, and hit RAM on T480s 8Gb
import pandas as pd

from ar6_trajectories import get_trajectories

from log_config import setup_logger
from utils import cache

setup_logger()
logger = logging.getLogger(__name__)


var_map = {
    "Population": "pop",
    "GDP|MER": "gdp",
    "Emissions|CO2": "co2",
    "Primary Energy": "tpec",
}
indicators = list(var_map.keys())

index_map = {
    "Variable": "variable",
    "Year": "year",
    "Region": "countrycode",
}


def _get_values_forward(group):
    _, df = group
    df = df.reset_index()
    df.set_index("Year", inplace=True)
    df["value_Y+5"] = df["value"].reindex(df.index + 5).values
    df["value_Y+10"] = df["value"].reindex(df.index + 10).values
    df["value_Y+15"] = df["value"].reindex(df.index + 15).values
    df["value_Y+20"] = df["value"].reindex(df.index + 20).values
    df["value_Y+25"] = df["value"].reindex(df.index + 25).values
    return df


def _shake(df):
    # Melt the dataframe from wide to long format
    result = df.reset_index()

    result = result.melt(
        id_vars=["Model", "Scenario", "Region", "Variable"],
        value_vars=result.columns,
        var_name="Year",
        value_name="value",
    )

    result = result.dropna()

    result["Year"] = result["Year"].astype("int16")
    result.set_index(["Model", "Scenario", "Region", "Variable", "Year"], inplace=True)

    with mp.Pool(mp.cpu_count()) as pool:
        group_data = list(result.groupby(["Model", "Scenario", "Region", "Variable"]))
        result = pd.concat(pool.map(_get_values_forward, group_data))

    result = result.reset_index()
    result.set_index(["Model", "Scenario", "Region", "Year", "Variable"], inplace=True)
    result.sort_index(inplace=True)
    logging.info(result)

    # Cleanup trajectories with NaNs and zeros
    result = result.dropna()
    result = result[(result != 0).all(axis=1)]

    # Align variables names mnemonics
    result = result.rename(index=var_map)
    # Align index level names
    result = result.rename_axis(index=index_map)

    return result


@cache(__file__)
@lru_cache(maxsize=128)
def get_sequences(n_samples=None):
    df_trajectories = get_trajectories()
    if n_samples:
        df_trajectories = df_trajectories[:n_samples]
    df = df_trajectories[
        df_trajectories.index.get_level_values("Variable").isin(indicators)
    ]
    df = df.drop(columns=["Unit"])
    df_sequences = _shake(df)
    return df_sequences


# sample = sequences(20000)

# %%


# When run directly, create the .pkl if necessary
if __name__ == "__main__":
    get_sequences()
