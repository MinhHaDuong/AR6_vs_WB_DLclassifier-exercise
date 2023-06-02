"""Find outliers in the table of IPCC AR6 WGIII scenarios

Outliers are values greater than twice the World 1990 value
Hopefully prints empty dataframes

Created on Tue May 23 10:09:14 2023
@author: haduong@centre-cired.fr
"""

from scipy.stats import zscore

from ar6_sequences import get_sequences

df = get_sequences()

z_scores = zscore(df)

THRESHOLD = 5

outliers = df[(z_scores > THRESHOLD).any(axis=1)]

print(outliers)
