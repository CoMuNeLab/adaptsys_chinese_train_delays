"""Compute and plot the correlation between delay and rain per station.

This script loads delay and rain data, computes Spearman correlation
coefficients for each station, and visualizes the results on a map.
The output is a PDF plot saved in the plots directory.
"""

import base
import numpy as np
import pandas as pd
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from matplotlib import patheffects as pe
from matplotlib import pyplot as plt
from scipy import stats


def load_delay():
    d = pd.read_csv("../data/delays_per_stations_h.csv.gz", index_col="time", parse_dates=["time"])
    return d.resample("1D").sum()


def load_rain():
    r = pd.read_csv("../data/rain_peaks_stations.csv.gz", index_col="time", parse_dates=["time"])
    return r.resample("1D").sum()


def main() -> None:
    """Do the main."""
    d = load_delay()
    print(d)
    r = load_rain().loc[d.index]
    print(r)

    nodes = base.load_nodes()
    nodes["spearman"] = [
        stats.spearmanr(d[station], r[station]).statistic for station in nodes.index
    ]
    nodes["pvalue"] = [stats.spearmanr(d[station], r[station]).pvalue for station in nodes.index]
    nodes = nodes.fillna({"spearman": 0.0, "pvalue": 1.0})

    fig, axs = plt.subplots(
        nrows=1, ncols=1, figsize=(5, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    for name, text in [
        ("Beijingxi", "Beijing"),
        ("Zhengzhou", "Zhengzhou"),
        ("Shanghaixi", "Shanghai"),
        ("Wuhan", "Wuhan"),
    ]:
        axs.annotate(
            text,
            (nodes.loc[name, "geometry"].x, nodes.loc[name, "geometry"].y),
            (nodes.loc[name, "geometry"].x, nodes.loc[name, "geometry"].y + 1),
            fontsize="x-small",
            arrowprops=dict(facecolor="black", lw=0.6, arrowstyle="-"),
            path_effects=[pe.withStroke(linewidth=3, foreground="w")],
        )

    base.load_graph(full=False).edges().plot(
        ax=axs, lw=1, alpha=0.2, rasterized=True, zorder=0.01, color="#999999"
    )
    axs.add_feature(cfeature.OCEAN, alpha=0.5)
    axs.add_feature(cfeature.BORDERS, linestyle="-", lw=0.2, alpha=0.2)
    nodes.plot(
        ax=axs,
        column="spearman",
        markersize=np.clip(-np.log10(nodes["pvalue"]), a_min=0, a_max=5) * 20,
        cmap="Spectral",
        vmin=-0.3,
        vmax=0.3,
        legend=True,
        legend_kwds={"label": "Spearman"},
        alpha=0.8,
        lw=0,
    )
    fig.tight_layout()
    fig.savefig("./plots/delay_rain_corr.pdf", dpi=300)


if __name__ == "__main__":
    main()
