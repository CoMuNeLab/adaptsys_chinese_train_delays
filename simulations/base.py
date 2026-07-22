"""Base lib.

This files contains common functions for all the main scripts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import diffsys
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from diffsys.models import Diffusion
from sklearn import model_selection

COPERNICUS = Path("../copernicus/")
TCACHE = Path("./stats/")
TCACHE.mkdir(parents=True, exist_ok=True)
CACHE = Path("../data/")
CACHE.mkdir(parents=True, exist_ok=True)
PLOTS = Path("./plots")
PLOTS.mkdir(parents=True, exist_ok=True)


LOC_CACHE = {}

pd.set_option("future.no_silent_downcasting", True)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_extfield() -> diffsys.ExternalField:
    """Load external field data from cache or Copernicus files.

    Returns
    -------
    diffsys.ExternalField
        The loaded external field data.
    """
    cachefile = Path("../data/extf.nc")
    if cachefile.is_file():
        return diffsys.ExternalField(xr.load_dataarray(cachefile, decode_coords="all"))
    efs = []
    _days = days()
    for fn in sorted((COPERNICUS / "CN:daily:total_precipitation").glob("*.nc")):
        log(f"Loading Ext Field: {fn}")
        _d = xr.load_dataarray(fn, decode_coords="all")
        _d = _d.sel(
            valid_time=np.isin(_d.valid_time.dt.date, np.asarray([d.date() for d in _days]))
        )
        if len(_d.valid_time) > 0:
            efs.append(_d)

    data = xr.concat(efs, dim="valid_time")
    data = data.fillna(0.0).rename({"valid_time": "time"}).squeeze().drop_vars("number")

    data.to_netcdf(cachefile)
    return diffsys.ExternalField(data)


def load_nodes() -> gpd.GeoDataFrame:
    """Load graph nodes from cache.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing graph nodes with station names as index.
    """
    return gpd.read_file(CACHE / "graph_nodes_metadata.geojson").set_index(
        "station_name_original", drop=True
    )


def load_graph(full: bool = True, days: pd.DatetimeIndex | None = None) -> diffsys.Graph:
    """Load the graph from cache.

    Parameters
    ----------
    full : bool, optional
        If True, all links are kept separated by month, weekday, hour.
        Otherwise, a global average is returned. Default is True.
    days : pd.DatetimeIndex or None, optional
        Specific days to filter transitions. Default is None.

    Returns
    -------
    diffsys.Graph
        The loaded graph with nodes and transitions.

    Notes
    -----
    The graph may contain duplicated links.
    """
    nodes = load_nodes()

    edges = gpd.read_file(CACHE / "graph.gpkg", layer="edges", rows=None)
    # Simmetrize
    edges = pd.concat(
        [edges, edges.rename(columns={"source": "target", "target": "source"})]
    ).drop_duplicates(subset=["source", "target"])
    edges = edges.set_index(["source", "target"], drop=True)

    transitions = pd.read_csv(CACHE / "aggregate_transitions.csv.gz", parse_dates=["date"])
    if days is not None:
        transitions = transitions[transitions["date"].isin(days)]

    transitions["month"] = transitions["date"].dt.month
    transitions["weekday"] = transitions["date"].dt.weekday
    transitions = transitions.drop(columns=["date"])
    # Remove self loops
    transitions = transitions[transitions["source"] != transitions["target"]]
    # filter out transitions not in edges
    tind = pd.MultiIndex.from_frame(transitions[["source", "target"]], names=["source", "target"])
    transitions = transitions.loc[tind.isin(edges.index)]

    tcount = pd.DataFrame(
        [
            {"val": 1, "m": d.month, "w": d.weekday()}
            for d in pd.date_range("2021-01-01", "2023-12-31")
        ]
    )
    if full:
        transitions["month"] = transitions["month"] == 8  # August
        transitions["weekday"] = transitions["weekday"] == 6  # Sunday
        transitions = (
            transitions.groupby(["source", "target", "month", "weekday", "hour"])
            .sum()
            .reset_index()
        )

        # Normalize by the number of time each combination appears
        tcount["m"] = tcount["m"] == 8
        tcount["w"] = tcount["w"] == 6
        tcount = tcount.groupby(["m", "w"]).sum()

        transitions["count"] = [
            t["count"] / tcount.loc[(t["month"], t["weekday"]), "val"]
            for _, t in transitions.iterrows()
        ]
    else:
        transitions = (
            transitions.drop(columns=["month", "weekday", "hour"])
            .groupby(["source", "target"])
            .sum()
            .reset_index()
        )
        # Normalize by the number of time each combination appears
        transitions["count"] /= len(tcount) * 24

    transitions["geometry"] = edges.loc[
        pd.MultiIndex.from_frame(transitions[["source", "target"]], names=["source", "target"]),
        "geometry",
    ].to_list()

    transitions = gpd.GeoDataFrame(transitions).set_crs(4326, allow_override=True, inplace=False)

    return diffsys.Graph(
        gpd.GeoDataFrame(nodes).set_crs(4326, allow_override=True, inplace=False),
        transitions,
        directed=False,
        edge_cols=["source", "target"],
    )


def params(
    path: list[Path] | Path | None = None, aggr=np.median, df: Literal["ci", "kfold"] | None = None
) -> dict[str, float] | pd.DataFrame:
    """Load the fitted parameters from cache.

    Parameters
    ----------
    path : list[Path] or Path or None, optional
        Path to parameter files. Default is None.
    aggr : callable, optional
        Aggregation function. Default is np.median.
    df : {"ci", "kfold"} or None, optional
        DataFrame type to return. Default is None.

    Returns
    -------
    dict[str, float] or pd.DataFrame
        Dictionary of parameters or DataFrame with confidence intervals/kfold results.

    Raises
    ------
    ValueError
        If the specified path does not exist.
    """
    if path is None:
        path = sorted((TCACHE / "optimize_pars").glob("*kfold-[0-9].jsonl*"))

    if isinstance(path, Path) and path.is_file():
        log(f"Loading {path}")
        pars = pd.read_json(path, lines=True)
        return pars.loc[pars["value"].idxmin(), ["alpha", "beta", "gamma"]].to_dict()

    if isinstance(path, list):
        pars = pd.DataFrame([params(p) for p in path])
        if df == "ci":
            out = []
            for _, vals in pars.items():
                # with that few values the CI are just the max and the min
                out.append({"low": np.min(vals), "high": np.max(vals), "stats": aggr(vals)})

            return pd.DataFrame(out, index=pars.columns)

        elif df == "kfold":
            pars.index = pd.Index([f"kf_{i}" for i in pars.index])
            return pars

        return pars.agg(aggr, axis=0).to_dict()
    raise ValueError(f"No such path {path}")


def log(*args, **kwargs) -> None:
    """Log a message with fancy formatting.

    Parameters
    ----------
    *args : tuple
        Positional arguments to log.
    **kwargs : dict
        Keyword arguments to log.
    """
    logger.info(*args, **kwargs)


def load_rain(local: bool | None = None, col: str = "tp"):
    """Load rain data from cache.

    Parameters
    ----------
    local : bool or None, optional
        If True, load station-specific rain data. Default is None.
    col : str, optional
        Column name to select. Default is "tp".

    Returns
    -------
    pd.DataFrame
        DataFrame with rain data indexed by time.
    """
    if local:
        return pd.read_csv(CACHE / "rain_peaks_stations.csv.gz", parse_dates=["time"]).set_index(
            "time"
        )
    return pd.read_csv(CACHE / "rain_peaks.csv.gz", parse_dates=["time"]).set_index("time")[[col]]


def load_peaks(year: int | list[int] | None = None, kind: str = "both") -> pd.DataFrame:
    """Load rain peaks data from cache.

    Parameters
    ----------
    year : int or list[int] or None, optional
        Year or list of years to filter. Default is None.
    kind : str, optional
        Type of peaks to load ("both", "high", "low", "full"). Default is "both".

    Returns
    -------
    pd.DataFrame
        DataFrame with rain peaks data.

    Raises
    ------
    NotImplementedError
        If an unsupported kind is specified.
    """
    data: pd.DataFrame = pd.read_csv(CACHE / "rain_peaks.csv.gz", index_col=0, parse_dates=True)
    if kind == "both":
        data = data.loc[data["peak"].isin(["high", "low"])]
    elif kind in {"high", "low"}:
        data = data.loc[data["peak"] == kind]
    elif kind == "full":
        data = data
    else:
        raise NotImplementedError()

    if year is not None:
        if isinstance(year, int):
            data = data.loc[data.index.year == year]  # type: ignore
        else:
            data = data.loc[data.index.year.isin(year)]  # type: ignore

    return data.sort_index()


def sim(model: Diffusion, full_graph: diffsys.Graph, usecache: bool = True) -> pd.DataFrame:
    """Simulate a day using the provided model.

    Parameters
    ----------
    model : Diffusion
        The diffusion model to use for simulation.
    full_graph : diffsys.Graph
        The graph containing all multiple links for each hour, month, day.
    usecache : bool, optional
        Whether to use cached rain data. Default is True.

    Returns
    -------
    pd.DataFrame
        The first and only cascade as a DataFrame.

    Notes
    -----
    Simulates one cascade. The model evolves through extreme events,
    generating delays and concluding the cascade.
    """
    global LOC_CACHE

    # Prepare cache for the rain.
    # Get the date
    day = model.ex_field.trange()[0]
    cache = CACHE / "rain_cache"
    cache.mkdir(parents=True, exist_ok=True)
    raincache = cache / f"rain_cache_{day.isoformat()[:10]}.csv.gz"
    if raincache.is_file() and usecache:
        event_cache = pd.read_csv(raincache, index_col=0)
        for c in event_cache.columns:
            LOC_CACHE[pd.Timestamp(c)] = event_cache[c].to_numpy()
    else:
        event_cache = {}

    for ev_hour in model.ex_field.extreme_events(kind="simple"):
        hour = ev_hour.trange()[0]
        gg: diffsys.Graph = full_graph.subset(
            [("weekday", hour.day_of_week == 6), ("month", hour.month == 8), ("hour", hour.hour)]
        ).drop_duplicates()

        model.evolve()
        if hour not in LOC_CACHE:
            gd = generated_delay(gg, ev_hour, trange=hour, weight="count")
            LOC_CACHE[hour] = gd
            event_cache[hour] = gd

        model.generate(LOC_CACHE[hour], "beta")
        model.generate(-model.graph.nodes()["capacity"].to_numpy(), "gamma")
        model.conclude_step(hour.to_datetime64(), threshold=0)
    model.conclude_cascade()

    if not raincache.is_file() and len(event_cache) > 1 and usecache:
        pd.DataFrame(event_cache, index=model.graph.nodes().index).to_csv(raincache)

    # Return the first and only cascade.
    return list(model.cascades())[0].df()


def generated_delay(
    graph: diffsys.Graph,
    external_field: diffsys.ExternalField,
    trange: pd.Timestamp | tuple[pd.Timestamp, pd.Timestamp] | None = None,
    weight: str | float = "weight",
    integral: pd.Series | None = None,
    **kwargs: float,
) -> np.ndarray:
    """Compute the generated delay before multiplying by the parameter.

    Parameters
    ----------
    graph : diffsys.Graph
        The graph to compute delays on.
    external_field : diffsys.ExternalField
        The external field affecting the graph.
    trange : pd.Timestamp or tuple[pd.Timestamp, pd.Timestamp] or None, optional
        Time range for the external field. Default is None.
    weight : str or float, optional
        Weight column name or value. Default is "weight".
    integral : pd.Series or None, optional
        Precomputed integral values. Default is None.
    **kwargs : float
        Additional keyword arguments for graph integration.

    Returns
    -------
    np.ndarray
        Array of generated delays.
    """
    if integral is None:
        if trange is None:
            trange = external_field.trange()
        integral = graph.integrate(trange=trange, ex_field=external_field, **kwargs)

    # Load the weight of each link (e.g. the number of trains going through).
    pass_count = graph.to_matrix(weight=weight)
    edge_weight = graph.to_matrix(weight=integral).multiply(pass_count)
    return edge_weight.sum(1).A.ravel()


def load_real_delay() -> pd.DataFrame:
    """Load real delay data from cache.

    Returns
    -------
    pd.DataFrame
        DataFrame with real delay data indexed by time.
    """
    return pd.read_csv(CACHE / "delays_per_stations.csv.gz", parse_dates=["time"], index_col="time")


def add_axis_label(ax, text: str):
    """Add a title to the axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to add the title to.
    text : str
        The title text.
    """
    ax.set_title(text, loc="left", fontsize="xx-large", fontweight="bold", ha="right")


def shorten_name(text: str) -> str:
    """Shorten a station name to a more compact format.

    Parameters
    ----------
    text : str
        The station name to shorten.

    Returns
    -------
    str
        The shortened station name.
    """
    parts = text.split()
    if parts[0] in {"san"}:
        return " ".join(parts[:2]).title() + " " + "".join([s[0].title() for s in parts[2:]])
    return parts[0].title() + " " + "".join([s[0].title() for s in parts[1:]])


def days(kf: int | None = None) -> pd.DatetimeIndex | tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Load days from cache and optionally split into train/test sets.

    Parameters
    ----------
    kf : int or None, optional
        K-fold index for splitting. If None, returns all days. Default is None.

    Returns
    -------
    pd.DatetimeIndex or tuple[pd.DatetimeIndex, pd.DatetimeIndex]
        If kf is None, returns all days as DatetimeIndex.
        Otherwise, returns tuple of (train_days, test_days).
    """
    days = pd.read_csv(CACHE / "delays_per_stations.csv.gz", parse_dates=["time"]).set_index(
        "time", drop=True
    )

    if kf is None:
        return days.index

    # kfold = model_selection.StratifiedKFold(n_splits=4, shuffle=True, random_state=1996)
    kfold = model_selection.KFold(n_splits=4, shuffle=True, random_state=1996)
    for i, (train_index, test_index) in enumerate(kfold.split(days.index)):
        if i == kf:
            train_days = days.index[train_index]
            test_days = days.index[test_index]
            break

    return train_days, test_days
