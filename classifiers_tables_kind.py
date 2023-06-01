#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:07:10 2023

@author: haduong
"""
import pandas as pd

from classifiers_compare import pretty_print
from classifiers_compare_kind import results


# %% Print and save the profiled results

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

with open("tables/classifiers_compare_kind.csv", "w", encoding="utf-8") as f:
    print(pretty_print(results, keys, "to_csv", sep="\t"), file=f)