""" Align the simulations dataframe: return a dataframe of 25-year sequences
Drop uninteresting variables
Cache result on disk because the input df_trajectories has 3.9 million rows.

Created on Tue May 23 12:57:57 2023

@author: haduong@centre-cired.fr
"""

import pickle
import pandas as pd
from ar6_trajectories import get_trajectories

FILENAME_CLEAN = "ar6_sequences.pkl"

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
    group = group.reset_index()
    group = group.set_index("Year")
    group = group.drop(columns=["Model", "Scenario", "Region", "Variable"])
    group["value_Y+5"] = group["value"].reindex(group.index + 5).values
    group["value_Y+10"] = group["value"].reindex(group.index + 10).values
    group["value_Y+15"] = group["value"].reindex(group.index + 15).values
    group["value_Y+20"] = group["value"].reindex(group.index + 20).values
    group["value_Y+25"] = group["value"].reindex(group.index + 25).values
    return group


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

    # Create the trajectories
    result = result.groupby(["Model", "Scenario", "Region", "Variable"]).apply(
        _get_values_forward
    )
    result = result.reorder_levels(["Model", "Scenario", "Region", "Year", "Variable"])
    result.sort_index(inplace=True)

    # Cleanup trajectories with NaNs and zeros
    result = result.dropna()
    result = result[(result != 0).all(axis=1)]

    # Align level names and variable labels
    result = result.rename(index=var_map)
    result = result.rename_axis(index=index_map)

    return result


# %%

try:
    df_sequences = pd.read_pickle(FILENAME_CLEAN)
    print("Success read  file", FILENAME_CLEAN)
except (IOError, EOFError, pickle.UnpicklingError) as e_read:
    print("Unable to access ", FILENAME_CLEAN, ":", e_read, ".")
    print("Attempting to create it.")
    try:
        df_trajectories = get_trajectories()
        df = df_trajectories[
            df_trajectories.index.get_level_values("Variable").isin(indicators)
        ]
        df = df.drop(columns=["Unit"])

        df_sequences = _shake(df)
        df_sequences.to_pickle(FILENAME_CLEAN)
        print("Cleaned OWID trajectories saved successfully!")
    except Exception as e:
        print("An error occurred while saving the OWID trajectories:", e)
