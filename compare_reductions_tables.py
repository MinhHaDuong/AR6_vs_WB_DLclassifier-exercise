"""Pretty print the result of the dimensionality reduction methods comparison results

Created on Tue June 1 13:46 2023

@author: haduong@centre-cired.fr
"""


import pandas as pd
from compare_reduction import get_results
from compare import pretty_print

results = get_results()

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 10000)

keys = ["normalized", "PCA", "latent", "latent2"]

print(pretty_print(results, keys, "to_string"))

with open("tables/compare_reductions.csv", "w", encoding="utf-8") as f:
    print(pretty_print(results, keys, "to_csv", sep="\t"), file=f)
