#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:34:14 2023

@author: haduong@centre-cired.fr
"""

import datetime
from powerset import get_results

result = get_results()

TAB = "\t"

message = f"""
Are IPCC AR6 scenarios realistic?

Author: haduong@centre-cired.fr
Run saved: {datetime.datetime.now()}

We define a sequence as a matrix with 6 columns and up to 4 rows.
The rows correspond to variables: CO2 emissions, GDP, populations, primary energy
The columns correspond to years, with a 5 years difference so that the trajectory is 25 years

The simulation sequences are picked from the IPCC AR6 national scenario database
The observations sequences are picked from owid-co2 dataset

We pool all the model, regions, years into two big sets of sequences,
and trained a GBM classifier to distinguish simulations from observations.

Results shows that
- Simulations are quite distinguishable from observations, even when looking at series in difference.
- Sequences with the 'tpec' variable are less distinguishable.
- Sequences with the 'population' variable are more distinguishable.
-> Simulations are realistic for energy, but the demographic dynamics is questionable.


{result}

Cut and paste-ready: Sorted by F1, tab-separated
{result.sort_values(by="F1", ascending=False).round(3).to_csv(sep=TAB)}
"""

print(message)
with open("tables/powerset.csv", "w", encoding="utf-8") as f:
    print(message, file=f)
