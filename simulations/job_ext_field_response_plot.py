"""Plot the response to constant external field."""

import base
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import optimize, stats


def fitting(x, a, b, c):
    return a * np.log(1 + np.exp(b + c * x))


def main() -> None:
    """Do the main."""
    response = pd.read_csv(base.CACHE / "ext_field_response.csv.gz", index_col=0).fillna(0)
    print(response)
    levels = np.asarray([float(x) for x in response.index]) * 1000
    tot = response.mean(1)

    boot = np.array(
        [
            tuple(stats.bootstrap((val,), statistic=np.mean).confidence_interval)
            for _, val in response.T.items()
        ]
    )

    fig, ax = plt.subplots()

    ax.fill_between(levels, boot[:, 0], boot[:, 1], zorder=0, alpha=0.2, color="C4", lw=0)
    ax.plot(levels, tot, "o", label="Prediction mean", color="C4", zorder=10)

    pars = optimize.curve_fit(fitting, levels, tot, p0=(1000, -10, 0.1), method="lm")[0]
    print(*pars, sep="\n")
    ax.plot(levels, fitting(levels, *pars), label=r"$A\ \log(1+e^{B+Cx})$", lw=2, color="C2")
    ax.plot(levels, levels * pars[2] * pars[0] + pars[1] * pars[0], label="Linear", lw=2)

    for c in response.columns:
        ax.plot(
            response.index * 1000,
            response[c],
            color="C7",
            lw=0.1,
            alpha=0.2,
            rasterized=True,
            zorder=0.001,
        )

    ax.set(
        ylim=(-20, 250),
        xlim=(-0.5, 6.5),
        xlabel="Stressor field intensity (mm)",
        ylabel="Cumulated delay (mins)",
    )
    ax.grid()
    ax.legend()

    fig.tight_layout()
    fig.savefig(base.PLOTS / "ext_field_response.pdf", dpi=300)
    fig.savefig(base.PLOTS / "ext_field_response.png", dpi=300)


if __name__ == "__main__":
    main()
