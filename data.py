"""Merge scenarios and observations, normalize and flatten.

The series in levels is transformed to a series where:
   - The first value is a fraction of the world in 1990
   - Subsequent values are factors from the previous value
   so that is A is the original and B the transformed vector
     a_i is the product from b_0  to  b_i

   This ensures that the values are around 1.


Created on Tue May  9 19:43:16 2023
@author: haduong@centre-cired.fr
"""

import numpy as np

simulations = np.load("simulations.npy")
observations = np.load("observations.npy")

labels = np.concatenate([
    np.zeros(len(simulations)),
    np.ones(len(observations))])

dataset = np.concatenate((simulations, observations))

world1990 = np.array([          # Source: ourworldindata.org
    27630,  # Emissions|CO2       Mt CO2/yr
    35850,  # GDP|MER             billion US$2010/yr
    5320,   # Population          million
    343.9])   # Primary Energy      EJ/yr,    95527 TWh

def as_change(arrays):
    """Convert a vector of levels into a vector of initial level and factors."""
    rotated = np.roll(arrays, 1, axis=2)
    rotated[:, :, 0] = 1
    return arrays / rotated

normalized = as_change(dataset)

normalized[:, :, 0] /= world1990

data =  normalized.reshape(normalized.shape[0], -1)

try:
    np.save('data.npy', data)
    print('Array data saved successfully!')
except Exception as e:
    print('An error occurred while saving the array data:', e)
   
try:
    np.save('labels.npy', labels)
    print('Array labels saved successfully!')
except Exception as e:
    print('An error occurred while saving the array labels:', e)