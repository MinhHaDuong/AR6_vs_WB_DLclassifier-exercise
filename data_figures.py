"""Examine scenarios and observations data.

Created on Tue May  9 19:43:16 2023
@author: haduong@centre-cired.fr
"""

from functools import lru_cache

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
    data_3d, LABELS[v] = get_data([v], diff=False, flatten=False)
    DATA[v] = flat(data_3d)
    DATA_DIFF[v] = flat(dif(data_3d))
    DATA_DIFF2[v] = flat(dif(dif(data_3d)))


# %% Display the sequences


def compare_data(axs, var="pop", diff=None, xlabel=None):
    if diff:
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
    if diff:
        titles = [s + " difference" for s in titles]

    for idx, (ax, matrix) in enumerate(zip(axs, [matrix1, matrix2])):
        lines = [list(zip(x, matrix[i, :])) for i in range(matrix.shape[0])]

        spaghettis = LineCollection(lines, linewidths=1, alpha=0.3)
        ax.add_collection(spaghettis)

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(matrix.min(), matrix.max())
        if diff:
            #            ax.set_ylim(0.5, 2)
            ax.axhline(0, color="black", linewidth=ax.spines["top"].get_linewidth())

        if xlabel:
            ax.set_xlabel("5 years period")
        if diff:
            ax.set_ylabel("Difference to next period")

        ax.set_title(titles[idx])
        ax.set_xticks(x.astype(int))

    axs[0].set_ylim(axs[1].get_ylim())


def fig_lines(diff=False, filename=None):
    _, axs = plt.subplots(4, 2, figsize=(12, 16))
    compare_data(axs[0, :], "co2", diff=diff)
    compare_data(axs[1, :], "pop", diff=diff)
    compare_data(axs[2, :], "gdp", diff=diff)
    compare_data(axs[3, :], "tpec", diff=diff, xlabel=True)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)


fig_lines(filename="figures/fig1-levels.png")
fig_lines(diff=True, filename="figures/fig2-changes.png")


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


@lru_cache(maxsize=None)  # Unbounded cache size
def compute_data(var, subsample_obs=3, subsample_sim=5):
    data = DATA[var]
    data_dif = DATA_DIFF[var]
    data_dif2 = DATA_DIFF2[var]
    labels = LABELS[var]

    num_obs = int(sum(labels))
    num_sim = int(len(labels) - num_obs)

    # Subsampling one of 5 or 3 sequence
    data_sim = data[0:num_sim][::subsample_sim, :]
    data_obs = data[num_sim:][::subsample_obs, :]
    data_sim_dif = data_dif[0:num_sim][::subsample_sim, :]
    data_obs_dif = data_dif[num_sim:][::subsample_obs, :]
    data_sim_dif2 = data_dif2[0:num_sim][::subsample_sim, :]
    data_obs_dif2 = data_dif2[num_sim:][::subsample_obs, :]

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
    _, axs = plt.subplots(4, 3, figsize=(18, 16))
    clouds(axs[0, :], "co2")
    clouds(axs[1, :], "pop")
    clouds(axs[2, :], "gdp")
    clouds(axs[3, :], "tpec")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)


fig_scatter("figures/fig3_2D.png")


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


fig_scatter3d([100], "figures/fig3_3D")

# %% Finetuning the viewpoint

# fig_scatter3d(range(90, 180, 10))

# %% Compare the CDFs


def plot_cdf(ax, data_obs, data_sim, var, x_label):
    """
    Plot the cumulative distribution function (CDF) for observed and simulated data.

    Parameters:
        ax (matplotlib.axes.Axes): The matplotlib Axes object to draw the plot on.
        data_obs (pandas.DataFrame): The observed data as a pandas DataFrame.
        data_sim (pandas.DataFrame): The simulated data as a pandas DataFrame.
        var (str): The variable name to be displayed in the plot legend.
        x_label (str): The column name in the data frames to be used as the x-axis values.

    Returns:
        None

    Warning: the CDF are not directly comparable because observations and simulation
    data were build by pooling from different sources ->
    Simulation data is biased towards large countries.
    """
    data_obs_sorted = np.sort(data_obs[x_label])
    data_sim_sorted = np.sort(data_sim[x_label])

    p_obs = 1.0 * np.arange(len(data_obs[x_label])) / (len(data_obs[x_label]) - 1)
    p_sim = 1.0 * np.arange(len(data_sim[x_label])) / (len(data_sim[x_label]) - 1)

    ax.plot(data_obs_sorted, p_obs, color="blue", alpha=0.5, label="observations")
    ax.plot(data_sim_sorted, p_sim, color="red", alpha=0.5, label="simulations")

    ax.set_ylabel("Cumulative Density")
    ax.legend(title=f"{var} {x_label.capitalize()}", loc="lower right")


def steps(axs, var):
    df_obs, df_sim = compute_data(var, subsample_obs=1, subsample_sim=1)
    plot_cdf(axs[0], df_obs, df_sim, var, "location")
    plot_cdf(axs[1], df_obs, df_sim, var, "trend")
    plot_cdf(axs[2], df_obs, df_sim, var, "variability")
    plot_cdf(axs[3], df_obs, df_sim, var, "acceleration")


def fig_cdf(filename=None):
    fig, axs = plt.subplots(4, 4, figsize=(18, 18))
    steps(axs[0, :], "co2")
    steps(axs[1, :], "pop")
    steps(axs[2, :], "gdp")
    steps(axs[3, :], "tpec")
    fig.suptitle(
        "Warning: Simulated data skewed towards larger countries compared to observed data.",
        fontsize=16,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    if filename:
        plt.savefig(filename)


fig_cdf("figures/fig4_cdf")
