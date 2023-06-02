#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:07:10 2023

@author: haduong
"""

import pandas as pd

from compare_classifiers import get_results
from compare import pretty_print

FILESTEM = "tables/compare_classifiers"

results = get_results()


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 10000)

keys = [
    "raw",
    "parallel raw",
    "normalized",
    "parallel normalized",
    "balanced",
    "parallel balanced",
    "base",
    "parallel base",
]

print(pretty_print(results, keys, "to_string"))

for key in keys:
    FILENAME = FILESTEM + "_" + key.replace(" ", "_") + ".tex"
    table = results.loc[key, "result"]
    table = table.iloc[:, :6]
    if "parallel" in key:
        table = table.drop(columns=["Train", "Predict"])
    table.to_latex(FILENAME, float_format="%.3f")
