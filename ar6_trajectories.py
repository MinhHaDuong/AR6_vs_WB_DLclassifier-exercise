"""
Access IPCC AR6 scenarios database of national results as a dataframe.

Expose scenario results as a Pandas dataframe.
The multiindex is Model - Scenario - Region - Variable.
Clean data to ensure consistent units.
Keep years at a five time step between 2005 and 2050.
Provide auxilliary functions to explore the list of variables.

References:
Database online at https://data.ece.iiasa.ac.at/ar6/#/docs
CSV data format at https://pyam-iamc.readthedocs.io/en/stable/data.html

Example use:
from ar6_trajectories import df_trajectories

Created on Thu Apr 28 2023
@author: haduong@centre-cired.fr
"""

import pickle
import pandas as pd

FILENAME_RAW = "AR6_Scenarios_Database_ISO3_v1.1.csv"
FILENAME_CLEAN = "ar6_trajectories.pkl"

# %% Import the data


def _get_dataframe(filename):
    coltypes = {
        "Model": "category",
        "Scenario": "category",
        "Region": "category",
        "Variable": "category",
        "Unit": "category",
        "2005": "float32",
        "2010": "float32",
        "2015": "float32",
        "2020": "float32",
        "2025": "float32",
        "2030": "float32",
        "2035": "float32",
        "2040": "float32",
        "2045": "float32",
        "2050": "float32",
    }

    return pd.read_csv(
        filename,
        index_col=[0, 1, 2, 3],
        usecols=[0, 1, 2, 3, 4, 8, 11, 16, 18, 23, 24, 25, 26, 27, 28],
        dtype=coltypes,
    )


def _check_units(data):
    """Verify if units were used consistently in the table.

    Return 0 if okay, otherwise list variables expressed in more than one unit,
    and return their number.
    """
    s = data["Unit"].droplevel([0, 1, 2]).groupby(level=0).unique()

    offending_rows = s[s.apply(lambda x: len(x) > 1)]

    if len(offending_rows):
        print(offending_rows)
    return len(offending_rows)


def _clean(df):
    """Fix units in IPCC AR6 scenarios database.

    Unfixed: MESSAGEix-GLOBIOM_1.0 report 0 GDP|MER in 2005 in lieu of nan
    """
    df["Unit"] = df["Unit"].str.replace("Million", "million")
    df["Unit"] = df["Unit"].str.replace("Int$2010", "US$2010")
    df["Unit"] = df["Unit"].str.replace("million full-time equivalent", "million")
    df.loc[df["Unit"] == "PJ/yr", "2005":"2050"] *= 0.001
    df["Unit"] = df["Unit"].str.replace("PJ/yr", "EJ/yr")
    df.loc[df["Unit"] == "million tkm", "2005":"2050"] *= 0.001
    df["Unit"] = df["Unit"].str.replace("million tkm", "bn tkm/yr")
    df.loc[df["Unit"] == "million pkm", "2005":"2050"] *= 0.001
    df["Unit"] = df["Unit"].str.replace("million pkm", "bn pkm/yr")

    # The unit 'US$2010/t or local currency' which is not unitary
    df = df.drop(df[df.index.get_level_values("Variable") == "Price|Carbon"].index)

    # Fix rows with 1000x error in population units
    scenarios = ["SSP2_1_75D-66", "SSP2_2D-66", "SSP2_BASE"]
    regions = ["CHN", "GBR", "KOR"]
    df.loc[
        (df.index.get_level_values("Model") == "TIAM-UCL 4.1.1")
        & (df.index.get_level_values("Scenario").isin(scenarios))
        & (df.index.get_level_values("Region").isin(regions))
        & (df.index.get_level_values("Variable") == "Population"),
        "2005":"2050",
    ] *= 0.001

    # Fix rows with 1000x error in GDP|MER units
    df.loc[
        (df.index.get_level_values("Model") == "GCAM-KAIST 1.0")
        & (df.index.get_level_values("Scenario").isin(["EN_NP_CurPol", "EN_NP_UNDC"]))
        & (df.index.get_level_values("Region") == "KOR")
        & (df.index.get_level_values("Variable") == "GDP|MER"),
        "2005":"2050",
    ] *= 0.001

    # Drop these rows, their error is not obviously 1000x
    df.drop(
        df[
            (
                df.index.get_level_values("Model").isin(
                    ["AIM/Enduse India 3.1", "India MARKAL", "MARKAL-India 1.0"]
                )
            )
            & (df.index.get_level_values("Region") == "IND")
            & (df.index.get_level_values("Variable") == "GDP|MER")
        ].index,
        inplace=True,
    )

    if _check_units(df):
        print("Alert: At least one Variable using more than one Unit.")

    return df


try:
    df_trajectories = pd.read_pickle(FILENAME_CLEAN)
    print("Successfully read cleaned AR6 trajectories from file", FILENAME_CLEAN)
except (IOError, EOFError, pickle.UnpicklingError) as e_read:
    print(
        "Unable to access ", FILENAME_CLEAN, ":", e_read, ".\nAttempting to create it."
    )
    try:
        df_trajectories = _clean(_get_dataframe(FILENAME_RAW))
        df_trajectories.to_pickle(FILENAME_CLEAN)
        print("Saved cleaned AR6 trajectories!")
    except Exception as e_write:
        print("An error occurred while saving the AR6 trajectories:", e_write)


def get_models(df):
    return df.index.get_level_values(0).unique().tolist()


def get_scenarios(df):
    return df.index.get_level_values(1).unique().tolist()


def get_regions(df):
    return df.index.get_level_values(2).unique().tolist()


def get_variables(df):
    return df.index.get_level_values(3).unique().tolist()


def get_years(df):
    return df.columns.tolist()[1:]


def filter_variables(df, substring):
    """Filter variable names containing a substring.

    Useful to find the indicator we want in the large number of variables.
    """
    variables = get_variables(df)
    return list(filter(lambda s: substring in s, variables))


def top_variables(df, n):
    """Count the most frequently reported variables."""
    count = df.groupby(level=3).size()
    variables = get_variables(df)
    return count[variables].sort_values(ascending=False)[0:n]


def root_variables(df):
    """List the variables at the root of the variables nomenclature and counts.

    This shows that some root variables are rarely reported.
    """
    variables = get_variables(df)
    roots = [s for s in variables if "|" not in s]
    return top_variables(df, -1)[roots].sort_values(ascending=False)
