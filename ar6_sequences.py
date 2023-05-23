#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:57:57 2023

@author: haduong@centre-cired.fr
"""

import pickle
import pandas as pd
from ar6_trajectories import df_trajectories

FILENAME_CLEAN = "ar6_sequences.pkl"

# %% Subsample the dataframe, too slow otherwise

indicators = [
    "Emissions|CO2",  # total net-CO2 emissions from all sources, Mt CO2/yr
    "GDP|MER",  # GDP at market exchange rate, billion US$2010/yr
    #    "GDP|PPP",
    "Population",
    "Primary Energy"  # ,
    #    "Secondary energy",
    #    "Final energy",
    #    "Capacity|Electricity",
    #    "Investment",
    #    "Consumption",
    #    "Land Cover|Cropland",
    #    "Land Cover|Pasture",
    #    "Land Cover|Forest",
]


def _get_values_forward(group):
    group = group.reset_index().set_index("Year")
    group = group.drop(columns=["Model", "Scenario", "Region", "Variable"])
    group["Value_Y+5"] = group["Value"].reindex(group.index + 5).values
    group["Value_Y+10"] = group["Value"].reindex(group.index + 10).values
    group["Value_Y+15"] = group["Value"].reindex(group.index + 15).values
    group["Value_Y+20"] = group["Value"].reindex(group.index + 20).values
    group["Value_Y+25"] = group["Value"].reindex(group.index + 25).values            
    return group


def _shake(df):
    # Melt the dataframe from wide to long format
    result = df.reset_index().melt(
        id_vars=["Model", "Scenario", "Region", "Variable"],
        value_vars=df.columns,
        var_name="Year",
        value_name="Value",
    ).dropna()
    
    result["Year"] = result["Year"].astype(int)
    result.set_index(["Model", "Scenario", "Region", "Variable", "Year"], inplace=True)

    # Create the trajectories
    result = result.groupby(["Model", "Scenario", "Region", "Variable"]).apply(_get_values_forward)
    result.drop(columns=['index'], inplace=True)
    result = result.reorder_levels(["Model", "Scenario", "Region", "Year", "Variable"]).sort_index()

    # Cleanup trajectories with NaNs and zeros
    result = result.dropna()
    result = result[(result != 0).all(axis=1)]

    return result

# %%

try:
    df_sequences = pd.read_pickle(FILENAME_CLEAN)
    print("Successfully read AR6 sequences from file", FILENAME_CLEAN)
except (IOError, EOFError, pickle.UnpicklingError) as e_read:
    print(
        "Unable to access ", FILENAME_CLEAN, ":", e_read, ".\nAttempting to create it."
    )
    try:
        df = df_trajectories[df_trajectories.index.get_level_values("Variable").isin(indicators)]
        df = df.drop(columns=["Unit"])
#        df = df.head(1000)   # For development
        
        df_sequences = _shake(df)
        df_sequences.to_pickle(FILENAME_CLEAN)
        print("Cleaned OWID trajectories saved successfully!")
    except Exception as e:
        print("An error occurred while saving the OWID trajectories:", e)
