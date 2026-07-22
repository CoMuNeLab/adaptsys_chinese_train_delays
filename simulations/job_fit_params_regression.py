"""Fit a regression model to predict delays from rainfall data.

This script performs Ridge regression to predict train delays based on
rainfall measurements. It computes correlation metrics between predicted
and actual delays and generates validation plots.
Output files include PDF and PNG plots saved in the plots directory.
"""

from pathlib import Path

import base
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import linear_model


def resample(fl: Path) -> pd.DataFrame:
    df = pd.read_csv(fl, index_col=0).T
    df.index = pd.DatetimeIndex(df.index)
    df = df.resample("1D").sum()
    return df


def load_peaks(kind: str = "both") -> pd.DataFrame:
    """Load the peaks."""
    data: pd.DataFrame = pd.read_csv("../data/rain_peaks.csv.gz", index_col=0, parse_dates=True)
    if kind == "both":
        data = data.loc[data["peak"].isin(["high", "low"])]
    elif kind in {"high", "low"}:
        data = data.loc[data["peak"] == kind]
    elif kind == "full":
        data = data
    else:
        raise NotImplementedError()

    data = data.rename(columns={"tp": "rain"}).resample("1D").sum()
    return data.sort_index()


def load_delay() -> pd.DataFrame:
    d = pd.read_csv(
        Path("../data/delays_per_stations.csv.gz"), index_col=0, parse_dates=[0]
    ).sort_index()
    d = (d - d.median(axis=0)).clip(lower=0.0)
    return d


# %%


def main() -> None:
    """Do the main."""
    train_days = base.days()
    test_days = base.days()
    rain = base.load_rain(col="mean").resample("1D").sum() * 1000
    test_rain = rain.loc[test_days]
    rain = rain.loc[train_days]

    print(rain.quantile([0, 0.25, 0.5, 0.75, 0.9, 1]))

    print(rain)
    delay = load_delay().sum(1)
    test_delay = delay.loc[test_days]
    delay = delay.loc[train_days]
    print(delay)
    print(test_delay)

    regr = linear_model.Ridge(fit_intercept=False)
    regr = regr.fit(rain, delay.to_frame())
    new = regr.predict(test_rain)
    results = pd.DataFrame(
        {"rain": test_rain["mean"], "real": test_delay / 60, "pred": new.squeeze() / 60}
    )
    print(results)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

    ax = axs
    results.plot.scatter(
        "pred",
        "real",
        c="rain",
        s=results["rain"] * 10 + 10,
        cmap="RdBu",
        alpha=0.6,
        ax=ax,
        vmin=1,
        vmax=10,
    )
    _, cax = fig.get_axes()
    cax.set(ylabel="Rainfall (mm)")
    ax.set(
        # xlim=(1900, 3500),
        # ylim=(-200, 3500),
        title="Ridge (CRN)",
        xlabel="Predicted delay (hours)",
        ylabel="Reported excess delay (hours)",
    )
    corr = stats.spearmanr(
        results[results["rain"] > 3]["pred"], results[results["rain"] > 3]["real"]
    )
    print(corr)
    # corr = stats.spearmanr(results["pred"], results["real"])
    # print(corr)
    ax.annotate(
        f"Spearman: {corr.statistic:3.2f}\np-value: {str(corr.pvalue)[:5] if corr.pvalue > 0.01 else '<0.01'}",
        (0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize="small",
        color="#666666",
        path_effects=[pe.withStroke(linewidth=2, foreground="w")],
    )

    ax.set_autoscale_on(False)
    xlim = ax.get_xlim()
    ax.plot(xlim, xlim, "k-.", alpha=0.5)

    fig.tight_layout()
    fig.savefig("./plots/validation_regression_2024.pdf", dpi=300)
    fig.savefig("./plots/validation_regression_2024.png", dpi=300)


if __name__ == "__main__":
    main()
