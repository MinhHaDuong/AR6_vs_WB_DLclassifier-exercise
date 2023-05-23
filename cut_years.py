"""Help to detect outliers.

Created on Tue May 23 10:09:14 2023
@author: haduong@centre-cired.fr
"""

import pandas as pd

# %% Generate the cut_years.csv
# On each line, a country and a year
# For sanity: Truncate the time series for the country on the given year
# and disregard data before

# To generate the dirty data you have to 
#  comment out the cleaning code in owid_trajectories.py
#  delete the (clean) file owid_trajectories.pkl
#  run owid_trajectories.py to regenerate it
#  rename the new file owid_trajectories.pkl as owid_trajectories-withoutliers.pkl
#  uncomment the cleaning code and regenerate as needed
df_trajectories = pd.read_pickle("owid_trajectories-withoutliers.pkl")

# Calculate the ratio of consecutive numbers
df_trajectories['ratio_Y+5'] = df_trajectories['value_Y+5'] / df_trajectories['value']
df_trajectories['ratio_Y+10'] = df_trajectories['value_Y+10'] / df_trajectories['value_Y+5']
df_trajectories['ratio_Y+15'] = df_trajectories['value_Y+15'] / df_trajectories['value_Y+10']
df_trajectories['ratio_Y+20'] = df_trajectories['value_Y+20'] / df_trajectories['value_Y+15']
df_trajectories['ratio_Y+25'] = df_trajectories['value_Y+25'] / df_trajectories['value_Y+20']

ratio_columns = ['ratio_Y+5', 'ratio_Y+10', 'ratio_Y+15', 'ratio_Y+20', 'ratio_Y+25']

outliers_df2 = df_trajectories[(df_trajectories[ratio_columns] > 2).any(axis=1)]
cut_years = outliers_df2.reset_index().groupby('country')['year'].max().dropna()

cut_years = cut_years.astype(int)
cut_years.to_csv('cut_years.csv', index=True)

del cut_years

# %% Verification

cut_years = pd.read_csv("cut_years.csv", index_col=0)
cut_years = cut_years[cut_years.columns[0]]

df2 = df_trajectories.reset_index()
for country, cut_year in cut_years.items():
    df2 = df2[~((df2['country'] == country) & (df2['year'] <= cut_year))]

assert df2[(df2[ratio_columns] > 2).any(axis=1)].empty

