"""Plot the distribution of the steady state."""

import json
from decimal import Decimal

import base
import diffsys
import geopandas as geopd
import h3pandas
import pandas as pd
from adjustText import adjust_text
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from matplotlib import axes
from matplotlib import pyplot as plt
from scipy import stats


def add_text(
    ax: axes.Axes,
    data: geopd.GeoDataFrame,
    xcol: str | None = None,
    ycol: str | None = None,
    **kwargs,
) -> None:
    kwargs.update({"lw": 0.1, "edgecolor": "k"})
    points = ax.scatter(
        data["geometry"].x if xcol is None else data[xcol],
        data["geometry"].y if ycol is None else data[ycol],
        **kwargs,
    )

    if xcol is not None and ycol is not None:
        to_label = pd.concat(
            [data.sort_values(by=xcol).tail(7), data.sort_values(by=ycol).tail(8)]
        ).drop_duplicates()
        text = [
            ax.annotate(
                str(lbl).title(), (_data[xcol], _data[ycol]), fontsize="xx-small", color="#444444"
            )
            for lbl, _data in to_label.iterrows()
        ]

        adjust_text(
            text,
            objects=points,
            prevent_crossings=True,
            force_text=(0.9, 1.5),
            force_pull=(0.001, 0.001),
            max_move=(10, 10),
            arrowprops=dict(arrowstyle="-", color="C5", alpha=0.5),
            ax=ax,
        )


# %%


def main() -> None:
    """Do the main."""
    graph = base.load_graph(full=False, days=base.days())
    nodes = geopd.read_file(base.CACHE / "ext_field_steady_state.geojson")
    nodes = nodes.set_index(nodes.columns[0]).sort_values("ss")
    centralities = geopd.read_file(base.CACHE / "coverage_data.geojson")
    centralities = centralities.set_index(centralities.columns[0])
    nodes = geopd.GeoDataFrame(
        pd.concat([nodes, centralities[["node_degree", "node_betweenness"]]], axis=1)
    )
    print(nodes)
    nodes["ss"] /= 60

    plot_steady_state_full(nodes, graph)
    # plot_steady_state(nodes, graph)
    # plot_steady_state_critical(nodes, graph)


def load_evolutions() -> pd.DataFrame:
    dt = []
    for f in sorted(base.CACHE.glob("*ext_field_steady_state_criticalpoint_dynamics_0*")):
        fname = f.name[:-7]

        stressor = Decimal(fname.split("_")[-1]) * 1000
        data = pd.read_csv(f, index_col="hours")
        dt.append(data["mean_delay"].rename(float(stressor)))

    return pd.concat(dt, axis=1).fillna(method="ffill")


def plot_steady_state_full(nodes: geopd.GeoDataFrame, graph: diffsys.Graph):
    fig = plt.figure(figsize=(10, 7))

    axs = fig.subplots(
        nrows=2,
        ncols=1,
        height_ratios=[1, 4],
        gridspec_kw={"hspace": 0, "wspace": 0, "left": 0.62, "right": 0.95},
        sharex="col",
    )
    _plot_steady_state_critical(axs[1], axs[0])
    fig.align_ylabels()
    base.add_axis_label(axs[0], "c")

    ax_maps = fig.subplots(
        ncols=2,
        nrows=1,
        sharey="row",
        gridspec_kw={"right": 0.52, "wspace": 0, "left": 0.08, "bottom": 0.45},
        subplot_kw={"projection": ccrs.PlateCarree(), "aspect": "auto"},
    )
    ax_scs = fig.subplots(
        ncols=2,
        nrows=1,
        sharey="row",
        gridspec_kw={"right": 0.52, "wspace": 0, "left": 0.08, "top": 0.4},
    )
    _plot_steady_state(
        nodes, graph, ax_scatter=ax_maps[0], ax_aggr=ax_maps[1], ax_deg=ax_scs[0], ax_bet=ax_scs[1]
    )
    base.add_axis_label(ax_maps[0], "a")
    base.add_axis_label(ax_scs[0], "b")

    # fig.tight_layout()
    fig.savefig(base.PLOTS / "steady_state_full.pdf")
    fig.savefig(base.PLOTS / "steady_state_full.png", dpi=300)


def plot_steady_state_critical(nodes: geopd.GeoDataFrame, graph: diffsys.Graph):
    fig, axs = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(6, 5),
        height_ratios=[1, 4],
        gridspec_kw={"hspace": 0, "wspace": 0},
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    _plot_steady_state_critical(axs[1, 0], axs[0, 0])

    fig.tight_layout()
    fig.savefig(base.PLOTS / "steady_state_critical.pdf")
    fig.savefig(base.PLOTS / "steady_state_critical.png", dpi=300)


def plot_steady_state(nodes: geopd.GeoDataFrame, graph: diffsys.Graph) -> None:
    fig, axs = plt.subplots(
        ncols=2,
        nrows=2,
        figsize=(7, 8),
        height_ratios=[3, 2],
        sharey="row",
        gridspec_kw={"wspace": 0},
    )

    _plot_steady_state(
        nodes, graph, ax_scatter=axs[0, 0], ax_aggr=axs[0, 1], ax_deg=axs[1, 0], ax_bet=axs[1, 1]
    )

    fig.tight_layout()
    fig.savefig(base.PLOTS / "steady_state.pdf")
    fig.savefig(base.PLOTS / "steady_state.png", dpi=300)


def _plot_steady_state_critical(ax_par: axes.Axes, ax_std: axes.Axes | None):
    ax_par.set_yscale("symlog", linthresh=1e-3)

    data = pd.read_csv(
        base.CACHE / "ext_field_steady_state_criticalpoint.csv.gz", index_col="stressor"
    )
    data.index = data.index * 1000
    with open("./stats/ext_field_steady_state_bounds.json", "rt") as fin:
        bounds = json.load(fin)

    approx_lower_threshold = bounds["lower"] * 1000
    approx_upper_threshold = bounds["upper"] * 1000
    lower_threshold = data[data["ss_delay"] <= 0].index[-1]
    upper_threshold = data["ss_delay"].index[-1]
    print(lower_threshold, upper_threshold)

    base.log(f"Theoretical thresholds: {approx_lower_threshold} -- {approx_upper_threshold}")
    base.log(f"Numerical thresholds:   {lower_threshold} -- {upper_threshold}")

    ax_par.plot(data.index, data["ss_delay"], ".-", lw=2, color="C4")
    ax_par.axvline(approx_lower_threshold, color="C6", ls=":", lw=3)
    ax_par.axvline(lower_threshold, color="C6", ls="-", label="Prop. threshold", lw=2)
    ax_par.fill_between(
        [0, lower_threshold],
        0,
        1,
        lw=0,
        color="C6",
        alpha=0.2,
        transform=ax_par.get_xaxis_transform(),
        label="Delay free",
    )
    ax_par.axvline(approx_upper_threshold, color="C7", ls=":", lw=3)
    ax_par.axvline(upper_threshold, color="C7", ls="-", label="Disr. threshold", lw=2)
    ax_par.fill_between(
        [upper_threshold, 50],
        0,
        1,
        lw=0,
        color="C7",
        alpha=0.2,
        transform=ax_par.get_xaxis_transform(),
        label="Delay explodes",
    )
    ax_par.set(
        xlabel="External stressor (mm)",
        ylabel="Average delay",
        xlim=(0, 1.5),
        ylim=(-1e-4, 1e2),
        yticks=[0, 1e-3, 1e-1, 1e1],
    )
    if ax_std:
        ax_std.fill_between(data.index, data["ss_delay_std"], lw=0, color="C4", alpha=0.5)
        ax_std.set(ylabel="std($s$)", title="Steady state", ylim=(0, 3), yticks=[3])

    ax_par.legend(
        fontsize="small",
        loc="lower left",
        bbox_to_anchor=((lower_threshold + upper_threshold) / 2, 0),
        bbox_transform=ax_par.transData,
    )
    arr_kw = {
        "length_includes_head": True,
        "lw": 0,
        "alpha": 0.3,
        "facecolor": "C4",
        "width": 0.02,
        "transform": ax_par.transAxes,
        "head_length": 0.04,
    }
    length = 0.15
    ax_par.arrow(0.08, 0.7, 0, -length, **arr_kw)
    ax_par.arrow(0.45, 0.3, 0, length, **arr_kw)
    ax_par.arrow(0.45, 0.7, 0, -length, **arr_kw)
    ax_par.arrow(0.9, 0.7 - length, 0, length, **arr_kw)


def _plot_steady_state(
    nodes: geopd.GeoDataFrame,
    graph: diffsys.Graph,
    ax_scatter: axes.Axes | None,
    ax_aggr: axes.Axes | None,
    ax_deg: axes.Axes | None,
    ax_bet: axes.Axes | None,
) -> None:
    if ax_scatter is not None:
        graph.edges().plot(
            ax=ax_scatter, lw=0.1, color="C1", alpha=0.2, rasterized=True, aspect=None, zorder=0
        )
        ax_scatter.scatter(
            nodes["geometry"].x,
            nodes["geometry"].y,
            s=nodes["ss"] * 1000,
            color="C1",
            edgecolor="k",
            lw=0.1,
            cmap="Reds",
            alpha=0.5,
        )
        bounds = nodes.total_bounds
        ax_scatter.set(
            title="Steady state", xlim=(bounds[0], bounds[2]), ylim=(bounds[1], bounds[3])
        )

    for ax in [ax_scatter, ax_aggr]:
        if ax is None:
            continue
        ax.add_feature(cfeature.BORDERS, alpha=0.5)
        ax.add_feature(cfeature.COASTLINE, alpha=0.5)
        ax.add_feature(cfeature.OCEAN, alpha=0.5, rasterized=True)

    if ax_aggr:
        nodes.h3.geo_to_h3_aggregate(3, operation="sum").plot(
            column="ss", ax=ax_aggr, cmap="Reds", aspect=None
        )
        bounds = nodes.total_bounds
        ax_aggr.set(title="Aggregation", xlim=(bounds[0], bounds[2]), ylim=(bounds[1], bounds[3]))

    if ax_deg:
        print("degree")
        print("SR", stats.spearmanr(nodes["node_degree"], nodes["ss"]))
        print("PR", stats.pearsonr(nodes["node_degree"], nodes["ss"]))
        add_text(
            ax_deg,
            nodes,
            "node_degree",
            "ss",
            s=nodes["node_betweenness"] * 1000,
            alpha=0.3,
            color="C2",
        )
        ax_deg.set(ylabel="Delay", xlabel="Degree", xlim=(-5, 55))

    if ax_bet:
        print("Betweenness")
        print("SR", stats.spearmanr(nodes["node_betweenness"], nodes["ss"]))
        print("PR", stats.pearsonr(nodes["node_betweenness"], nodes["ss"]))
        add_text(
            ax_bet, nodes, "node_betweenness", "ss", s=nodes["node_degree"], alpha=0.3, color="C4"
        )
        ax_bet.set(xlabel="Betweenness", xlim=(-0.04, 0.64))
        if ax_deg is None:
            ax_bet.set_ylabel(
                "Delay"
                # ylim=(-300 / 60, 3800 / 60)
            )


if __name__ == "__main__":
    main()
