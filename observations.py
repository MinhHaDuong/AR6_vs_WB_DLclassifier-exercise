"""
Access the Our World In Data (OWID) as a dataframe.

See OWID data pipeline https://docs.owid.io/projects/etl/en/latest/api/python/

Created on Thu Apr 20 15:03:33 2023
@author: haduong
"""


import pandas as pd
import numpy as np
from owid_trajectories import df


def _select(df, indicators):
    """Return a masked dataframe with rows to include the variables in the indicators list.

    For any pair (country, year),
    the result contains the row (country, v, year) for any v in indicators
    the result does not contains any other row
    the result is the largest such subdataframe.
    """
    assert not df.empty, "The dataframe to select from must not be empty."
    assert indicators, "Indicators must be a non-empty list."
    
    mask = pd.Series(
        df.index.get_level_values('variable').isin(indicators),
        index=df.index)
    group_counts = mask.groupby(level=[0, 1]).transform('sum')
    mask[group_counts != len(indicators)] = False

    result = df[mask]

    assert set(result.index.get_level_values('variable').unique()) == set(indicators), "Incorrect Variables selection."
    assert not result.empty, "The result is empty. Too many indicators?"

    return result


def get_observations(indicators):
    subdf = _select(df, indicators)

    result = np.array([
        a
        for _, a in subdf.groupby(level=[0, 1])
        if not (np.isnan(a).any() | (a == 0).any()).any()
        ])
    return result
