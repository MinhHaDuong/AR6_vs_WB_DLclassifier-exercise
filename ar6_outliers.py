"""Find outliers in the table of IPCC AR6 WGIII scenarios

Outliers are values greater than twice the World 1990 value
Hopefully prints empty dataframes

Created on Tue May 23 10:09:14 2023
@author: haduong@centre-cired.fr
"""

import pandas as pd

from data import world_1990
from ar6_sequences import df_sequences as df


threshold = 2
floor = 0.0001

df = df.reset_index()

world_1990_reset = world_1990.reset_index()
world_1990_reset.columns = ["Variable", "unit"]

df = pd.merge(df, world_1990_reset, on="Variable", how='left')

columns_to_divide = ['Value', 'Value_Y+5', 'Value_Y+10', 'Value_Y+15', 'Value_Y+20', 'Value_Y+25']

for col in columns_to_divide:
    df[col] = df[col] / df['unit']

df = df.set_index(["Model", "Scenario", "Region", "Year", 'Variable'], drop=True)  # Add the other index levels as needed

df.drop(columns="unit", inplace=True)

outliers = df[df.gt(threshold).any(axis=1)]
print(outliers)

values2010 = df.xs(2010, level="Year")["Value"]
outliers = values2010[values2010 < floor]
print(outliers)

# To check the values are really outliers
# df_sub = df.xs('Global TIMES 2.0', level="Model")
# q = df.xs(('EPPA 6', "gdp"), level=["Model", "Variable"])
# q = df.xs(('IMACLIM-NLU 1.0', "co2"), level=["Model", "Variable"])
# q = df.xs(('EU', 2010, "co2"), level=["Region", "Year", "Variable"])