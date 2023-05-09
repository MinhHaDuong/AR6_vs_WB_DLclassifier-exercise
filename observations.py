"""
Access the Our World In Data (OWID) as a dataframe.

See OWID data pipeline https://docs.owid.io/projects/etl/en/latest/api/python/

Created on Thu Apr 20 15:03:33 2023
@author: haduong
"""

import pandas as pd
from owid import catalog


# %%

pd.set_option('display.max_columns', None)

df = catalog.find('owid_co2').load()

df = df[['co2', 'gdp', 'population', 'primary_energy_consumption']]
