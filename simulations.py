"""Access the ar6 scenario DataFrame, return an array of arrays

Created on Thu Apr 28 2023

@author: haduong@centre-cired.fr
"""

from sequence2array import sequence2array
from ar6_sequences import df_sequences


def get_simulations(indicators, units=1):
    return sequence2array(
        df_sequences,
        indicators,
        units,
        "Variable",
        ["Model", "Scenario", "Region", "Year"],
    )


# def _select(df, indicators):
#     """Return a masked dataframe with rows to include the variables in the indicators list.

#     For any pair (country, year),
#     the result contains the rows (country, year, v) for any v in indicators
#     the result does not contains any other row
#     the result is the largest such subdataframe of df
#     """
#     assert indicators, "Indicators must be a non-empty list."
#     assert not df.empty, "DataFrame to select from must not be empty."
#     assert not df.isna().any().any(), "DataFrame must not contains NaN values"
#     assert not (df == 0).any().any(), "DataFrame must not contains zero values"

#     mask = pd.Series(
#         df.index.get_level_values("Variable").isin(indicators), index=df.index
#     )
#     group_counts = mask.groupby(
#         level=["Model", "Scenario", "Region", "Year"]
#     ).transform("sum")
#     mask[group_counts != len(indicators)] = False

#     result = df[mask].sort_index()

#     assert set(result.index.get_level_values("Variable").unique()) == set(
#         indicators
#     ), "Incorrect Variables selection."
#     assert (
#         not result.empty
#     ), "The result is empty. Match the list of indicators to observations data."

#     return result


# def get_simulations(indicators, units=1):
#     subdf = _select(df_sequences, indicators)
#     subdf = subdf.div(units, level="Variable", axis=0)

#     result = np.array(
#         [a for _, a in subdf.groupby(level=["Model", "Scenario", "Region", "Year"])]
#     )
#     return result
