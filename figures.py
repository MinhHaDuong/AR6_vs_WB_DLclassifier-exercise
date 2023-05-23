"""Examine scenarios and observations data.

Created on Tue May  9 19:43:16 2023
@author: haduong@centre-cired.fr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from data import get_data

# %% Verify the data: well normalized, no outliers, ...


def compare_data(axs, sim="Population", obs="population", as_change=None, xlabel=None):
    data, labels = get_data([sim], [obs], as_change)

    num_obs = int(sum(labels))
    num_sim = int(len(labels) - num_obs)

    data_sim = data[0:num_sim][::5, :]
    data_obs = data[num_sim:][::3, :]

    matrix1 = data_obs
    matrix2 = data_sim

    x = np.arange(matrix1.shape[1])

    titles = [obs + " observations", sim + " simulations"]

    for idx, (ax, matrix) in enumerate(zip(axs, [matrix1, matrix2])):
        lines = [list(zip(x, matrix[i, :])) for i in range(matrix.shape[0])]

        lc = LineCollection(lines, linewidths=1, alpha=0.3)
        ax.add_collection(lc)

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(matrix.min(), matrix.max())
        if as_change:
            ax.set_ylim(0.5, 2)
            ax.axhline(1, color="black", linewidth=ax.spines["top"].get_linewidth())

        # Set labels
        if xlabel:
            ax.set_xlabel("5 years period")
        if as_change:
            ax.set_ylabel("Change from previous period")
        else:
            ax.set_ylabel("Fraction of world 1990")
            
        ax.set_title(titles[idx])
        ax.set_xticks(x.astype(int))

    axs[0].set_ylim(axs[1].get_ylim())


def fig_lines(as_change=False, filename=None):
    _, axs = plt.subplots(4, 2, figsize=(12, 16))
    compare_data(axs[0, :], "Emissions|CO2", "co2", as_change=as_change)
    compare_data(axs[1, :], "Population", "population", as_change=as_change)
    compare_data(axs[2, :], "GDP|MER", "gdp", as_change=as_change)
    compare_data(
        axs[3, :],
        "Primary Energy",
        "primary_energy_consumption",
        as_change=as_change,
        xlabel=True,
    )
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


fig_lines(filename="fig1-levels.png")
fig_lines(as_change=True, filename="fig2-changes.png")

# %%


def compute_data(sim, obs):
    # Compute for as_change=False
    data, labels = get_data([sim], [obs])
    data_change, _ = get_data([sim], [obs], as_change=True)

    num_obs = int(sum(labels))
    num_sim = int(len(labels) - num_obs)

    data_sim = data[0:num_sim][::5, :]
    data_obs = data[num_sim:][::3, :]

    data_sim_change = data_change[0:num_sim][::5, :]
    data_obs_change = data_change[num_sim:][::3, :]


    df_obs = pd.DataFrame({
        'levels_mean': data_obs.mean(axis=1),
        'levels_std': data_obs.std(axis=1),
        'change_mean': data_obs_change.mean(axis=1),
        'change_std': data_obs_change.std(axis=1)
    })

    df_sim = pd.DataFrame({
        'levels_mean': data_sim.mean(axis=1),
        'levels_std': data_sim.std(axis=1),
        'change_mean': data_sim_change.mean(axis=1),
        'change_std': data_sim_change.std(axis=1)
    })

    return df_obs, df_sim


def plot_data(ax, data_obs, data_sim, obs, sim, x_label, y_label, hline=None):
    # Plot mean versus standard deviation for observations and simulations
    ax.scatter(data_obs[x_label], data_obs[y_label], color='blue', alpha=0.3,
               label=obs + ' observations')
    ax.scatter(data_sim[x_label], data_sim[y_label], color='red', alpha=0.1,
               label=sim + ' simulations')

    ax.set_xlabel(x_label.capitalize())
    ax.set_ylabel(y_label.capitalize())
    if hline:
        ax.axhline(1, color="black", linewidth=ax.spines["top"].get_linewidth())

    ax.legend()


def clouds(axs, sim, obs):
    df_obs, df_sim = compute_data(sim, obs)

    plot_data(axs[0], df_obs, df_sim, obs, sim, 'levels_mean', 'levels_std')
    plot_data(axs[1], df_obs, df_sim, obs, sim, 'levels_mean', 'change_mean', hline=True)
    plot_data(axs[2], df_obs, df_sim, obs, sim, 'levels_mean', 'change_std')
    plot_data(axs[3], df_obs, df_sim, obs, sim, 'levels_std', 'change_mean', hline=True)
    plot_data(axs[4], df_obs, df_sim, obs, sim, 'levels_std', 'change_std')
    plot_data(axs[5], df_obs, df_sim, obs, sim, 'change_std', 'change_mean', hline=True)


def fig_scatter(filename=None):
    fig, axs = plt.subplots(4, 6, figsize=(36, 16))
    clouds(axs[0, :], "Emissions|CO2", "co2")
    clouds(axs[1, :], "Population", "population")
    clouds(axs[2, :], "GDP|MER", "gdp")
    clouds(axs[3, :], "Primary Energy", "primary_energy_consumption")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

fig_scatter("fig3_multiple.png")
