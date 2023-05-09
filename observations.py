"""
Access the Our World In Data (OWID) as a dataframe.

See OWID data pipeline https://docs.owid.io/projects/etl/en/latest/api/python/

Created on Thu Apr 20 15:03:33 2023
@author: haduong
"""

import pandas as pd
from owid import catalog
import world_bank_data as wb


# %%

pd.set_option('display.max_columns', None)

df = catalog.find('owid_co2').load()

df = df[['co2', 'gdp', 'population', 'primary_energy_consumption']]

# %% Use the World Bank Data module at https://pypi.org/project/world-bank-data/.


codeWB = {
    'Emissions|CO2': 'EN.ATM.CO2E.KT',
    'GDP|MER': 'NY.GDP.MKTP.KD',            # GDP (constant 2015 US$)
    'Land Cover|Forest': 'AG.LND.FRST.K2',  # Use  AG.LND.FRST.ZS  for % of land area
    'Population': 'SP.POP.TOTL',
    'Population|Urban': 'SP.URB.TOTL.IN.ZS',  # As %. Use SP.URB.TOTL for total
    'Primary Energy': 'NA'}


def getWB(variable):
    s = wb.get_series(codeWB[variable], date='1960:2022')
    return s.dropna()


dfWB = pd.concat([getWB('Emissions|CO2'), getWB('GDP|MER'), getWB('Population')])

"""The World Bank energy data leaves to be desired:
series Primary Energy use (kg of oil equivalent per capita) ('EG.USE.PCAP.KG.OE')
ends in 2014
series Energy intensity level of primary energy (MJ/$2017 PPP GDP) (EG.EGY.PRIM.PP.KD)
ends in 2019
I did not find Total Primary Energy, or any Secondary Energy statistics.
"""

# %% Example calls to the API
# pd.set_option('display.max_rows', 6)
# topics = wb.get_topics()
# print(topics)

# sources = wb.get_sources()
# print(sources)

# countries = wb.get_countries()
# print(countries)

# regions = wb.get_regions()
# print(regions)

# incomelevels = wb.get_incomelevels()
# print(incomelevels)

# lendingtypes = wb.get_lendingtypes()
# print(lendingtypes)
