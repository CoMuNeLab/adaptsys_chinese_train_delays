"""Study the system response to varying external field levels.

This script simulates the response of the train network to different
levels of external field stress. It computes cumulative delays for each
node at multiple stressor levels using a diffusion model.
Output is a CSV file with response data saved in the cache directory.
"""

import base
import diffsys
import numpy as np
import pandas as pd
import xarray as xr
from diffsys.models import Diffusion
from tqdm.contrib import concurrent

graph_adj = base.load_graph(full=False, days=base.days()).drop_duplicates()
graph_tmp = base.load_graph(full=True, days=base.days())
LOC_CACHE = {}


def exfield_like(exfiel: diffsys.ExternalField, fill_values: float):
    """Make a new external stressor with the same coords but filled by a fixed value."""
    data = xr.DataArray(
        np.full(exfiel.shape, fill_values), coords=exfiel.data.coords
    )  # From 50mm it's heavy rain
    return diffsys.ExternalField(data)


def sim_all(data: tuple[float, diffsys.ExternalField, pd.Series, dict]):
    global graph_adj
    global graph_tmp

    level, stressor, rain, params = data

    if level == 0:
        return pd.DataFrame({f"{level:5.4f}": []})

    mod = Diffusion(graph_adj, level * stressor, **params)
    for exf in mod.ex_field.extreme_events(kind="simple"):
        hour = exf.trange()[0]

        mod.evolve()

        if hour not in LOC_CACHE:
            LOC_CACHE[hour] = base.generated_delay(
                graph_tmp.subset([("weekday", False), ("month", False), ("hour", hour.hour)]),
                exf,
                trange=hour,
                weight="count",
                integral=rain,
            )
        stressor = level * LOC_CACHE[hour]
        mod.generate(stressor, "beta")
        mod.generate(-mod.graph.nodes()["capacity"].to_numpy(), "gamma")
        mod.conclude_step(hour.to_datetime64(), threshold=0)

    mod.conclude_cascade()

    # Transform to DataFrame
    delays = list(mod.cascades())[0].df()
    if len(delays) == 0:
        return pd.DataFrame({f"{level:5.4f}": [], "node": []}).set_index("node", drop=True)

    return (
        delays.drop(columns=["failing", "time"])
        .groupby("node")
        .sum()
        .rename(columns={"value": f"{level:5.4f}"})
        .copy(deep=True)
    )


def main() -> None:
    """Do the main."""
    # Load the network
    global graph_adj
    global graph_tmp

    # build the stressor
    stressor_levels = np.linspace(0.0, 0.01, 31)
    stressor = exfield_like(base.load_extfield().get(day="2020-01-01"), fill_values=1.0)

    # Params
    params = base.params()
    base.log(params)

    rain = graph_adj.integrate(stressor, trange=pd.Timestamp("2020-01-01 08:00:00"), ds=1.0)

    delays = concurrent.process_map(
        sim_all,
        [(sl, stressor, rain, params) for sl in stressor_levels],
        max_workers=8,
        chunksize=1,
    )

    delays_df = pd.concat(delays, ignore_index=False, axis=1)
    delays_df.T.to_csv(base.CACHE / "ext_field_response.csv.gz")
    print(delays_df)


if __name__ == "__main__":
    main()

# %%
