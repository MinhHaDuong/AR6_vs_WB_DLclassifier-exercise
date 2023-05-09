"""Normalize and merge scenarios and observations

Created on Tue May  9 19:43:16 2023
@author: haduong@centre-cired.fr
"""


from observations import observations


# %% Normalize

"""The series in levels is transformed to a series where:
   - The first value is a fraction of the world in 1990
   - Subsequent values are factors from the previous value
   so that is A is the original and B the transformed vector
     a_i is the product from b_0  to  b_i

   This ensures that the values are around 1.
"""


def as_change(a):
    """Convert a vector of levels into a vector of initial level and factors."""
    b = np.zeros_like(a)
    b[0] = a[0]
    for i in range(1, len(a)):
        if abs(a[i - 1]) <= 0.001:
            print("Warning: division by ", a[i-1], " at i=", i)
        b[i] = a[i] / a[i - 1]
    return b


normalized = np.apply_along_axis(as_change, 2, simulations)

world1990 = np.array([          # Source: ourworldindata.org
    27630,  # Emissions|CO2       Mt CO2/yr
    35850,  # GDP|MER             billion US$2010/yr
    5320,   # Population          million
    343.9])   # Primary Energy      EJ/yr,    95527 TWh

normalized[:, :, 0] /= world1990
