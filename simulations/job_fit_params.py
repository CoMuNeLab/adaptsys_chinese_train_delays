"""Fit the parameter alpha, beta and gamma to the learning dataset.

This script needs to be run multiple times changing the value of KFOLD from 0 to 3 manually.
"""

import json
import logging
from functools import partial

import base
import diffsys
import numpy as np
import pandas as pd
import xarray as xr
from base import TCACHE
from diffsys.models import Diffusion
from scipy import optimize
from tqdm.contrib.concurrent import process_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NJOBS = 1
KFOLD = 3
base.log(f"Fitting the {KFOLD} kfold.")

# %%


train_days, test_days = base.days(kf=KFOLD)

# %%

REAL_DELAY = pd.DataFrame({})
GRAPH_ADJ = diffsys.Graph.empty()
GRAPH_TMP = diffsys.Graph.empty()
EF = diffsys.ExternalField.empty()


def prepare_data(train_days=train_days, kfold=KFOLD):
    global REAL_DELAY
    global GRAPH_ADJ
    global GRAPH_TMP
    global EF
    REAL_DELAY = base.load_real_delay()
    logger.info(f"Loading Files for {len(REAL_DELAY)} days")
    baseline = REAL_DELAY.loc[train_days].median()
    REAL_DELAY = (REAL_DELAY - baseline).clip(lower=0.0)

    GRAPH_ADJ = base.load_graph(full=False, days=train_days).drop_duplicates()
    logger.info(f"Full graph: {GRAPH_ADJ}")
    GRAPH_ADJ._nodes["delay_q50"] = baseline

    GRAPH_TMP = base.load_graph(full=True, days=train_days)
    logger.info(f"Graph {GRAPH_TMP}")

    EF = base.load_extfield()
    print(f"ExternalField: {EF}")


prepare_data()

# %%


def _simulate(pars: tuple, days: pd.DatetimeIndex):
    if len(pars) == 2:
        alpha = 1.0
        beta, gamma = pars
    elif len(pars) == 3:
        alpha, beta, gamma = pars
    else:
        raise ValueError()

    logger.info(f"Using:: α={alpha}, β={beta}, γ={gamma}")
    func = partial(base.sim, full_graph=GRAPH_TMP, usecache=True)
    cascades = process_map(
        func,
        [
            Diffusion(
                GRAPH_ADJ, EF.get(day=peak_day), alpha=alpha, beta=beta, gamma=gamma, weight="count"
            )
            for peak_day in days
        ],
        total=len(days),
        max_workers=NJOBS,
        chunksize=5,
    )
    return cascades


def _test_cascasdes(cascades: list[pd.DataFrame]):
    excess_delay = REAL_DELAY
    delta = 0
    n = 0

    daily_rain = EF.data.resample(time="D").sum()

    # Save position in `DataArray` to use as point coordinates
    st_lons = xr.DataArray([p.x for p in GRAPH_ADJ.nodes()["geometry"]])
    st_lats = xr.DataArray([p.y for p in GRAPH_ADJ.nodes()["geometry"]])
    for cascade, day in zip(cascades, test_days):
        rainy = daily_rain.sel(
            longitude=st_lons,
            latitude=st_lats,
            time=xr.DataArray([day] * len(st_lats)),
            method="nearest",
        )

        # Find stations where there was at least a bit of rain
        rainy_stats = [s for s, r in zip(GRAPH_ADJ.nodes().index, rainy.data) if r > 0]
        if len(cascade) == 0:
            delta_v = excess_delay.fillna(0.0)

            n += 1
        else:
            delta_v = (
                cascade.set_index("node", drop=True)["value"] - excess_delay.loc[day]
            ).fillna(0.0)

        delta += np.sum(np.power(np.abs(delta_v[rainy_stats].to_numpy()), 2.0))
        # delta += np.sum(np.power(np.abs(delta_v.to_numpy()), 2.0))

    logger.info(f"Got Δ={delta:g} (zeros = {n} / {len(test_days)})")
    return delta


# %%


def simulate(pars: tuple):
    cascades = _simulate(pars, days=test_days)
    delta = _test_cascasdes(cascades)

    if len(pars) == 3:
        data = {k: x for k, x in zip(["alpha", "beta", "gamma"], pars)}
    else:
        data = {k: x for k, x in zip(["beta", "gamma"], pars)}
        data["alpha"] = 1.0
    data["value"] = delta

    folder = TCACHE / "optimize_pars"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"kfold-{KFOLD}.jsonl").open("a") as fout:
        json.dump(data, fout, separators=(",", ":"))
        fout.write("\n")
        fout.flush()

    return delta


# %%


def main() -> None:
    """Do the main."""
    prepare_data()
    pars = (1.01, 9, 0.05)
    optimize.minimize(
        simulate, pars, bounds=[(1.0, 100.0), (0, 1000.0), (0, 100.0)], method="Nelder-Mead"
    )


if __name__ == "__main__":
    main()
