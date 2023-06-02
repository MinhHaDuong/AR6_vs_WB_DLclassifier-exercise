"""Pretty print the result of the dimensionality reduction methods comparison results

Created on Tue June 1 13:46 2023

@author: haduong@centre-cired.fr
"""

import pandas as pd
from compare_reductions import get_results
from compare import pretty_print

FILESTEM = "tables/compare_reductions"

results = get_results()

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 10000)

keys = ["normalized", "PCA", "latent", "latent2"]

print(pretty_print(results, keys, "to_string"))

for key in keys:
    FILENAME = FILESTEM + "_" + key.replace(" ", "_") + ".tex"
    table = results.loc[key, "result"]
    table = table.iloc[:, 2:6]
    if key in ["PCA", "latent2"]:
        KEEP_INDEX = False
    else:
        KEEP_INDEX = True
    table.to_latex(FILENAME, float_format="%.3f", index=KEEP_INDEX)
