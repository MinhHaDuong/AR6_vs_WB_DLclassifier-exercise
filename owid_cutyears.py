"""Generate the file owid_cutyears.csv used to sanitize OWID CO2 dataset

On each line, a country and a year
Truncate the time series for the country on the given year
and disregard data before

Created on Tue May 23 10:09:14 2023
@author: haduong@centre-cired.fr
"""

import pandas as pd
from owid_sequences import FILENAME_RAW, get_dataframe, shake

FILENAME_CUTYEARS = "owid_cutyears.csv"


def _assess_cut_years():
    df_sequences = shake(get_dataframe(FILENAME_RAW))

    # Calculate the ratio of consecutive numbers
    df_sequences["ratio_Y+5"] = df_sequences["value_Y+5"] / df_sequences["value"]
    df_sequences["ratio_Y+10"] = df_sequences["value_Y+10"] / df_sequences["value_Y+5"]
    df_sequences["ratio_Y+15"] = df_sequences["value_Y+15"] / df_sequences["value_Y+10"]
    df_sequences["ratio_Y+20"] = df_sequences["value_Y+20"] / df_sequences["value_Y+15"]
    df_sequences["ratio_Y+25"] = df_sequences["value_Y+25"] / df_sequences["value_Y+20"]

    ratio_columns = [
        "ratio_Y+5",
        "ratio_Y+10",
        "ratio_Y+15",
        "ratio_Y+20",
        "ratio_Y+25",
    ]

    outliers_df2 = df_sequences[(df_sequences[ratio_columns] > 2).any(axis=1)]
    cut_years = outliers_df2.reset_index().groupby("country")["year"].max().dropna()

    cut_years = cut_years.astype(int)
    return cut_years


try:
    cut_years = pd.read_csv(FILENAME_CUTYEARS, index_col=0)
    cut_years = cut_years[cut_years.columns[0]]
    print("Successfully read cut years trajectories from file", FILENAME_CUTYEARS)
except (IOError, EOFError) as e_read:
    print(
        "Unable to access ",
        FILENAME_CUTYEARS,
        ":",
        e_read,
        ".\nAttempting to create it.",
    )
    try:
        cut_years = _assess_cut_years()
        cut_years.to_csv(FILENAME_CUTYEARS, index=True)
        print("Cleaned OWID trajectories saved successfully!")
    except Exception as e:
        print("An error occurred while saving the OWID trajectories:", e)
