"""
Read the Our World In Data (OWID) CO2 dataset.
Select variables, change units, build 25-year trajectories with 5 year timestep
Expose as a Pandas DataFrame with multindex [country, variable, year]
and columns [value, value_Y+5, ..., value_Y+25]
Cache the result in a .pkl file

See OWID data pipeline https://docs.owid.io/projects/etl/en/latest/api/python/

Usage: from owid_trajectories import df_trajectories

Created on Thu Apr 20 15:03:33 2023
@author: haduong@centre-cired.fr
"""

import pickle
import pandas as pd

# Got it at  https://github.com/owid/co2-data/blob/master/owid-co2-data.csv
# Units at https://github.com/owid/co2-data/blob/master/owid-co2-codebook.csv
FILENAME_RAW = "owid-co2-data.csv"
FIlENAME_CLEAN = "owid_trajectories.pkl"


def _get_dataframe(filename):
    # Note: We would prefer to use nullable integers than floats but Pandas 1.x has only NaN floats.
    coltypes = {
        "country": "category",
        "year": "int",
        "population": "float32",
        "gdp": "float32",
        "co2": "float32",
        "primary_energy_consumption": "float32",
    }

    units = {
        "population": 1e6,  # Population in Million
        "gdp": 1e9,  # GDP in billion  $ (international 2011)
        "co2": 1,  # CO2 Mt
        "primary_energy_consumption": 277.8,
    }  # Primary energy in Ej from TWh

    df = pd.read_csv(
        filename, index_col=[0, 1], usecols=coltypes.keys(), dtype=coltypes
    ).div(pd.Series(units))
    
    todrop = [
        'World',
        'Africa',
        'Africa (GCP)',
        'Asia',
        'Asia (GCP)',
        'Asia (excl. China and India)',
        'Central America (GCP)',
        'Europe',
        'Europe (GCP)',
        'Europe (excl. EU-27)',
        'Europe (excl. EU-28)',
        'European Union (27)',
        'European Union (27) (GCP)',
        'European Union (28)',
        'French Equatorial Africa (GCP)',
        'French Equatorial Africa (Jones et al. 2023)',
        'French West Africa (GCP)',
        'French West Africa (Jones et al. 2023)',
        'International transport',
        'High-income countries',
        'Kuwaiti Oil Fires (GCP)',
        'Kuwaiti Oil Fires (Jones et al. 2023)',
        'Least developed countries (Jones et al. 2023)',
        'Leeward Islands (GCP)',
        'Leeward Islands (Jones et al. 2023)',
        'Low-income countries',
        'Lower-middle-income countries',
        'Middle East (GCP)',
        'Non-OECD (GCP)',
        'North America',
        'North America (GCP)',
        'North America (excl. USA)',
        'OECD (GCP)',
        'OECD (Jones et al. 2023)',
        'Oceania',
        'Oceania (GCP)',
        'Panama Canal Zone (GCP)',
        'Panama Canal Zone (Jones et al. 2023)',
        'Ryukyu Islands (GCP)',
        'Ryukyu Islands (Jones et al. 2023)',
        'South America',
        'South America (GCP)',
        'St. Kitts-Nevis-Anguilla (GCP)',
        'St. Kitts-Nevis-Anguilla (Jones et al. 2023)',
        'Upper-middle-income countries']

    df = df[~df.index.get_level_values('country').isin(todrop)]
    
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


def _shake(df):
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
    df_trajectories = pd.read_pickle(FIlENAME_CLEAN)
    print("Successfully read OWID trajectories from file", FIlENAME_CLEAN)
except (IOError, EOFError, pickle.UnpicklingError) as e_read:
    print(
        "Unable to access ", FIlENAME_CLEAN, ":", e_read, ".\nAttempting to create it."
    )
    try:
        df_trajectories = _shake(_get_dataframe(FILENAME_RAW))
        df_trajectories.to_pickle(FIlENAME_CLEAN)
        print("Cleaned OWID trajectories saved successfully!")
    except Exception as e:
        print("An error occurred while saving the OWID trajectories:", e)
