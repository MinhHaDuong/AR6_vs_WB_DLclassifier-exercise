"""Access the ar6 scenario DataFrame, return an array of arrays

Created on Thu Apr 28 2023

@author: haduong@centre-cired.fr
"""

import pandas as pd
import numpy as np
from ar6_trajectories import df_trajectories
# from ar6_sequences import df_sequences

# %% Keep only the rows describing the indicators


def _select(df, indicators):
    """Return a masked dataframe with rows to include the variables in the indicators list.

    For any triple (model, scenario, region),
    the result contains the row (model, scenario, region, v) for any v in indicators
    the result does not contains any other row
    the result is the largest such subdataframe.
    """
    assert not df.empty, "The dataframe to select from must not be empty."
    assert indicators, "Indicators must be a non-empty list."

    mask = pd.Series(
        df.index.get_level_values("Variable").isin(indicators), index=df.index
    )
    group_counts = mask.groupby(level=[0, 1, 2]).transform("sum")
    mask[group_counts != len(indicators)] = False

    result = df[mask]

    assert set(result.index.get_level_values("Variable").unique()) == set(
        indicators
    ), "Incorrect Variables selection."
    assert len(result["Unit"].unique()) == len(indicators), "More units than variables."
    assert not result.empty, "The result is empty. Too many indicators?"

    return result


# %% Make a numpy array of 25-years trajectories


def get_simulations(indicators, units=1):
    subdf = _select(df_trajectories, indicators).drop(columns=["Unit"])
    subdf = subdf.div(units, level="Variable", axis=0)

    width = 6  # At five years step, twenty five years including extremities
    num_trajectories = df_trajectories.shape[1] - width
    result = np.array(
        [
            a[:, i : i + width]
            for _, trajectory in subdf.groupby(level=["Model", "Scenario", "Region"])
            for a in [trajectory.values]
            for i in range(num_trajectories)
            if not (np.isnan(a) | (a == 0)).any()
        ]
    )
    return result

# %% New code, the df_sequence

def _select_2(df, indicators):
    """Return a masked dataframe with rows to include the variables in the indicators list.

    For any pair (country, year),
    the result contains the rows (country, year, v) for any v in indicators
    the result does not contains any other row
    the result is the largest such subdataframe of df
    """
    assert indicators, "Indicators must be a non-empty list."
    assert not df.empty, "DataFrame to select from must not be empty."
    assert not df.isna().any().any(), "DataFrame must not contains NaN values"
    assert not (df == 0).any().any(), "DataFrame must not contains zero values"

    mask = pd.Series(
        df.index.get_level_values("Variable").isin(indicators), index=df.index
    )
    group_counts = mask.groupby(level=["Model", "Scenario", "Region", "Year"]).transform("sum")
    mask[group_counts != len(indicators)] = False

    result = df[mask]

    assert set(result.index.get_level_values("Variable").unique()) == set(
        indicators
    ), "Incorrect Variables selection."
    assert (
        not result.empty
    ), "The result is empty. Match the list of indicators to observations data."

    return result


def get_simulations_2(indicators, units=1):
    subdf = _select_2(df_sequences, indicators)
    subdf = subdf.div(units, level="Variable", axis=0)

    result = np.array([a for _, a in subdf.groupby(level=["Model", "Scenario", "Region", "Year"])])
    return result
