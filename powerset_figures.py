#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:34:14 2023

@author: haduong@centre-cired.fr
"""

import matplotlib.pyplot as plt
from powerset import get_results

result = get_results()

def graph(score_name):
    plt.clf()
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    y0 = result[score_name].min() * 0.99

    for variable, ax in zip(["co2", "gdp", "pop", "tpec"], axes.flatten()):
        mask = result.index.str.contains(variable)
        true_values = result[mask][score_name]
        false_values = result[~mask][score_name]
        values = [true_values, false_values]

        ax.boxplot(values, positions=[1, 2])
        ax.set_xticklabels([f"{variable} present\n", f"{variable} absent\n"])
        ax.tick_params(axis="x", which="both", length=0)

        for i, value in enumerate(values, start=1):
            y = value
            x = [i] * len(y)
            ax.plot(x, y, "r.", alpha=0.8)

        ax.set_ylim(y0, 1)

    plt.suptitle(
        f"Effect of the presence of a variable on the classifier performance ({score_name} score)"
    )
    plt.tight_layout()
    plt.savefig(f"figures/single_variable_{score_name}.png")

graph("AUC")
graph("F1")
