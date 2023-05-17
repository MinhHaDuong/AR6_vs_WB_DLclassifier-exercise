#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:34:14 2023

@author: haduong@centre-cired.fr
"""

import itertools
import pandas as pd

import matplotlib.pyplot as plt

from classify import classify

from data import indicators_simulations, indicators_observations

result = pd.DataFrame(columns=['Test loss', 'Accuracy'])
result.index.name='Variables'

pairs = list(zip(indicators_simulations, indicators_observations))

for r in range(1, len(pairs) + 1):
    # Generate and print all subsets of size r
    for subset in itertools.combinations(pairs, r):
        isim, iobs = zip(*subset)
        print(list(isim), list(iobs))
        score = classify(list(isim), list(iobs))
        key = str(iobs)
        result.loc[key] = score
        
for v in indicators_observations:
    result[v] = [v in i for i in result.index]
   

result = result.reset_index().set_index(indicators_observations)

result.to_csv('realism.csv')

# %%

sorted_result = result.sort_values(by='Accuracy', ascending=False)
print(sorted_result.to_string(index=False))

with open('sorted_result.txt', 'w') as f:
    print(sorted_result.to_string(index=False), file=f)

# %%

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

series = result['Accuracy']

for variable, ax in zip(indicators_observations, axes.flatten()):
    # Filter the series for when the variable is True and False
    true_values = series.xs(True, level=variable)
    false_values = series.xs(False, level=variable)

    values = [true_values, false_values]
    ax.boxplot(values, labels=['In', 'Out'])
    
    ax.set_title(f'{variable}')
    ax.set_ylabel('Accuracy')
    ax.legend()

# Adjust spacing between subplots
plt.tight_layout()

plt.savefig('Accuracy_comparison.png')

# Clear the current plot to prepare for the next iteration
plt.clf()

# Show the plot
plt.show()

# %%
series = result['Test loss']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

for variable, ax in zip(indicators_observations, axes.flatten()):
    # Filter the series for when the variable is True and False
    true_values = series.xs(True, level=variable)
    false_values = series.xs(False, level=variable)

    values = [true_values, false_values]
    ax.boxplot(values, labels=['In', 'Out'])
    
    ax.set_title(f'{variable}')
    ax.set_ylabel('Test loss')
    ax.legend()

# Adjust spacing between subplots
plt.tight_layout()

plt.savefig('Test_loss_comparison.png')

# Clear the current plot to prepare for the next iteration
plt.clf()

# Show the plot
plt.show()
