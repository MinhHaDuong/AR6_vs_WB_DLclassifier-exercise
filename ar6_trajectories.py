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

import logging
import pickle
import pandas as pd

from log_config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

FILENAME_RAW = "AR6_Scenarios_Database_ISO3_v1.1.csv"
FILENAME_CLEAN = "ar6_trajectories.pkl"

# %% Import the data


def _get_dataframe(filename=FILENAME_RAW):
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

    # Fix rows with 1000x error in tpec units
    df.loc[
        (df.index.get_level_values("Model") == "Global TIMES 2.0")
        & (df.index.get_level_values("Variable") == "Primary Energy"),
        "2005":"2050",
    ] *= 0.001

    # Drop rows those error is not obviously fixable.
    # Drop EU not a country
    targets = [
        [("AIM/Enduse India 3.1", "IND", "GDP|MER"), ("Model", "Region", "Variable")],
        [("India MARKAL", "IND", "GDP|MER"), ("Model", "Region", "Variable")],
        [("MARKAL-India 1.0", "IND", "GDP|MER"), ("Model", "Region", "Variable")],
        [("EPPA 6", "2CNow_OptTax", "GDP|MER"), ("Model", "Scenario", "Variable")],
        [("IMACLIM-NLU 1.0", "Emissions|CO2"), ("Model", "Variable")],
        [("NEMESIS 5.0", "MLT", "Emissions|CO2"), ("Model", "Region", "Variable")],
        [("EU"), ("Region")],
    ]

    for key, levels in targets:
        rows_to_drop = df.xs(key, level=levels, drop_level=False).index
        df.drop(rows_to_drop, inplace=True)

    if _check_units(df):
        print("Alert: At least one Variable using more than one Unit.")

    return df


def get_trajectories():
    try:
        df_trajectories = pd.read_pickle(FILENAME_CLEAN)
        logging.info(f"   Success read file {FILENAME_CLEAN}")
        return df_trajectories
    except (IOError, EOFError, pickle.UnpicklingError) as e_read:
        logging.info(f"Unable to fetch {FILENAME_CLEAN} : {e_read} .")
        logging.info("Attempting now to create it.")
    try:
        df_trajectories = _clean(_get_dataframe())
        df_trajectories.to_pickle(FILENAME_CLEAN)
        logging.info("Saved cleaned AR6 trajectories!")
        return df_trajectories
    except Exception as e_write:
        logging.error(
            f"   An error occurred while saving the AR6 trajectories: {e_write} "
        )
    return None


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


# When run directly, create the .pkl if necessary 
if __name__ == "__main__":
    get_trajectories()

