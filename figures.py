"""Examine scenarios and observations data.

Created on Tue May  9 19:43:16 2023
@author: haduong@centre-cired.fr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from data import get_data, dif, flat, all_vars

# Load the data

DATA = {}
DATA_DIFF = {}
DATA_DIFF2 = {}
LABELS = {}

for v in all_vars:
    data_3d, labels = get_data([v], as_change=False, flatten=False)
    DATA[v] = flat(data_3d)
    DATA_DIFF[v] = flat(dif(data_3d))
    DATA_DIFF2[v] = flat(dif(dif(data_3d)))
    LABELS[v] = labels


# %% Display the sequences


def compare_data(axs, var="pop", as_change=None, xlabel=None):
    if as_change:
        data = DATA_DIFF[var]
    else:
        data = DATA[var]

    labels = LABELS[var]
    num_obs = int(sum(labels))
    num_sim = int(len(labels) - num_obs)

    data_sim = data[0:num_sim][::5, :]
    data_obs = data[num_sim:][::3, :]

    matrix1 = data_obs
    matrix2 = data_sim

    x = np.arange(matrix1.shape[1])

    titles = [var + " observations", var + " simulations"]
    if as_change:
        titles = [s + " difference" for s in titles]

    for idx, (ax, matrix) in enumerate(zip(axs, [matrix1, matrix2])):
        lines = [list(zip(x, matrix[i, :])) for i in range(matrix.shape[0])]

        lc = LineCollection(lines, linewidths=1, alpha=0.3)
        ax.add_collection(lc)

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(matrix.min(), matrix.max())
        if as_change:
            #            ax.set_ylim(0.5, 2)
            ax.axhline(0, color="black", linewidth=ax.spines["top"].get_linewidth())

        if xlabel:
            ax.set_xlabel("5 years period")
        if as_change:
            ax.set_ylabel("Difference to next period")
        else:
            ax.set_ylabel("Fraction of world 1990")

        ax.set_title(titles[idx])
        ax.set_xticks(x.astype(int))

    axs[0].set_ylim(axs[1].get_ylim())


def fig_lines(as_change=False, filename=None):
    _, axs = plt.subplots(4, 2, figsize=(12, 16))
    compare_data(axs[0, :], "co2", as_change=as_change)
    compare_data(axs[1, :], "pop", as_change=as_change)
    compare_data(axs[2, :], "gdp", as_change=as_change)
    compare_data(axs[3, :], "tpec", as_change=as_change, xlabel=True)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


fig_lines(filename="fig1-levels.png")
fig_lines(as_change=True, filename="fig2-changes.png")


# %% Display 2D scatterplots


def residuals(data):
    """
    Compute the residuals of a numpy array after fitting a line
    from the first to the last point.

    Parameters
    ----------
    data : numpy.ndarray
        The input data.

    Returns
    -------
    numpy.ndarray
        The residuals after subtracting the fitted line.
    """
    x = np.array([0, len(data) - 1])
    y = np.array([data[0], data[-1]])
    coefficients = np.polyfit(x, y, 1)
    line = np.poly1d(coefficients)
    x_full_range = np.arange(len(data))
    y_full_range = line(x_full_range)
    return data - y_full_range


def compute_data(var):
    data = DATA[var]
    data_dif = DATA_DIFF[var]
    data_dif2 = DATA_DIFF2[var]
    labels = LABELS[var]

    num_obs = int(sum(labels))
    num_sim = int(len(labels) - num_obs)

    # Subsampling one of 5 or 3 sequence
    data_sim = data[0:num_sim][::5, :]
    data_obs = data[num_sim:][::3, :]

    data_sim_dif = data_dif[0:num_sim][::5, :]
    data_obs_dif = data_dif[num_sim:][::3, :]
    data_sim_dif2 = data_dif2[0:num_sim][::5, :]
    data_obs_dif2 = data_dif2[num_sim:][::3, :]

    data_sim_residuals = pd.DataFrame(map(residuals, data_sim))
    data_obs_residuals = pd.DataFrame(map(residuals, data_obs))

    df_obs = pd.DataFrame(
        {
            "location": data_obs.mean(axis=1),
            "variability": data_obs_residuals.std(axis=1),
            "trend": data_obs_dif.mean(axis=1),
            "acceleration": data_obs_dif2.mean(axis=1),
        }
    )

    df_sim = pd.DataFrame(
        {
            "location": data_sim.mean(axis=1),
            "variability": data_sim_residuals.std(axis=1),
            "trend": data_sim_dif.mean(axis=1),
            "acceleration": data_sim_dif2.mean(axis=1),
        }
    )

    return df_obs, df_sim


def plot_data(ax, data_obs, data_sim, var, x_label, y_label, vline=None, hline=None):
    ax.scatter(
        data_sim[x_label],
        data_sim[y_label],
        color="red",
        alpha=0.1,
        label=var + " simulations",
    )
    ax.scatter(
        data_obs[x_label],
        data_obs[y_label],
        color="blue",
        alpha=0.3,
        label=var + " observations",
    )
    ax.set_xlabel(x_label.capitalize())
    ax.set_ylabel(y_label.capitalize())
    if vline:
        ax.axvline(0, color="black", linewidth=ax.spines["top"].get_linewidth())
    if hline:
        ax.axhline(0, color="black", linewidth=ax.spines["top"].get_linewidth())

    ax.legend()


def clouds(axs, var):
    df_obs, df_sim = compute_data(var)
    plot_data(axs[0], df_obs, df_sim, var, "location", "trend", hline=True)
    plot_data(
        axs[1], df_obs, df_sim, var, "trend", "acceleration", vline=True, hline=True
    )
    plot_data(axs[2], df_obs, df_sim, var, "location", "variability")


def fig_scatter(filename=None):
    fig, axs = plt.subplots(4, 3, figsize=(18, 16))
    clouds(axs[0, :], "co2")
    clouds(axs[1, :], "pop")
    clouds(axs[2, :], "gdp")
    clouds(axs[3, :], "tpec")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


fig_scatter("fig3_2D.png")


# %% Display a 3D scatterplot


def cloud3d(ax, df_obs, df_sim, var, azimut):
    ax.scatter(
        df_sim["location"],
        df_sim["trend"],
        df_sim["acceleration"],
        color="red",
        alpha=0.01,
    )
    ax.scatter(
        df_obs["location"],
        df_obs["trend"],
        df_obs["acceleration"],
        color="blue",
        alpha=0.3,
    )
    ax.set_xlabel("location")
    ax.set_ylabel("trend")
    ax.set_zlabel("acceleration")
    ax.view_init(elev=30, azim=azimut)
    ax.set_title(f"{var}")


def fig_scatter3d(azimuths, filename=None):
    fig = plt.figure(figsize=(3 * len(azimuths), 16))
    variables = ["co2", "pop", "gdp", "tpec"]
    for i, var in enumerate(variables):
        df_obs, df_sim = compute_data(var)
        for j, azim in enumerate(azimuths):
            ax = fig.add_subplot(
                4, len(azimuths), i * len(azimuths) + j + 1, projection="3d"
            )
            cloud3d(ax, df_obs, df_sim, var, azimut=azim)
    if filename:
        plt.savefig(filename)
    plt.show()


fig_scatter3d([100], "fig3_3D")

# %% Finetuning the viewpoint

# fig_scatter3d(range(90, 180, 10))
