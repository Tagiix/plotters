"""Gallery cards for distributions and the correlogram."""

import numpy as np
import pandas as pd

import plotters as pl

from . import datasets as ds
from .registry import card


def _lognormals() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {"sigma 0.5": rng.lognormal(0, 0.5, 4000), "sigma 1.5": rng.lognormal(0, 1.5, 4000)}
    )


def _ar1(n=900, phi=0.6) -> pd.Series:
    rng = np.random.default_rng(6)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return pd.Series(x, name="AR(1) phi=0.6")


def _arch(n=1500) -> pd.Series:
    rng = np.random.default_rng(7)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = np.sqrt(0.2 + 0.7 * x[t - 1] ** 2) * rng.normal()
    return pd.Series(x, name="ARCH(1)")


@card(
    "Distributions",
    "plot_null_test",
    [
        "Histogram of the null; both tails beyond the 2.5% quantiles shaded in the warm colour.",
        "Solid 'observed' line; annotation at the top right gives p, alpha and the verdict.",
    ],
)
def dist_null():
    null = np.random.default_rng(8).normal(size=20_000)
    return pl.plot_null_test(
        null, 2.3, "Is the statistic extreme?", "p counted from the simulated distribution"
    )


@card(
    "Distributions",
    "plot_null_test: one-sided",
    [
        "Only the left tail is shaded and the observed value sits inside it.",
    ],
)
def dist_null_left():
    return pl.plot_null_test(
        np.random.default_rng(9).normal(size=20_000), -2.0, "Left tail", tail="left"
    )


@card(
    "Distributions",
    "plot_box",
    [
        "Ranked by median, smallest at the top; one hue; the long-tailed group shows outliers.",
        "Figure height grows with the number of groups.",
    ],
)
def dist_box():
    rng = np.random.default_rng(10)
    frame = pd.DataFrame(
        {
            f"model {c}": rng.lognormal(m, s, 120)
            for c, m, s in [
                ("A", 0.0, 0.3),
                ("B", 0.2, 0.5),
                ("C", -0.1, 0.9),
                ("D", 0.4, 0.2),
                ("E", 0.1, 0.6),
            ]
        }
    )
    return pl.plot_box(frame, "Error by model", "ranked by median", reference=1.0, points=True)


@card(
    "Distributions",
    "plot_box: log value axis",
    [
        "Log value axis with readable ticks so a tail decades past the box is still visible.",
    ],
)
def dist_box_log():
    return pl.plot_box(_lognormals(), "Heavy tails", "logvalue=True", logvalue=True)


@card(
    "Distributions",
    "plot_histogram",
    [
        "Two overlaid histograms on shared bins with dashed normal fits in matching colours.",
    ],
)
def dist_hist():
    return pl.plot_histogram(
        ds.returns(2), "Overlaid distributions", normal=True, density=True, bins=50
    )


@card(
    "Distributions",
    "plot_ecdf: survival",
    [
        "Complementary ECDF on a log y axis: the heavier tail decays more slowly.",
    ],
)
def dist_ecdf():
    return pl.plot_ecdf(
        _lognormals(), "Tails", "P(X >= x), log y", complementary=True, logx=True, logy=True
    )


@card(
    "Distributions",
    "plot_qq: normal",
    [
        "Points bend away from the dotted reference at both ends on fat-tailed data.",
    ],
)
def dist_qq_normal():
    return pl.plot_qq(ds.returns(1), "Normal QQ", "fat tails bend the ends")


@card(
    "Distributions",
    "plot_qq: Student-t",
    [
        "Same data against a fitted Student-t: points hug the line; the fitted nu is in the subtitle.",
    ],
)
def dist_qq_t():
    return pl.plot_qq(ds.returns(1), "Student-t QQ", dist="t")


@card(
    "Distributions",
    "plot_cumulative_share",
    [
        "Curves sit above the 'paid evenly' diagonal; the heavier-tailed series is more concentrated.",
    ],
)
def dist_share():
    return pl.plot_cumulative_share(_lognormals(), "Concentration of a total")


@card(
    "Distributions",
    "plot_expectile_curve",
    [
        "Both curves pass through 1 at tau = 0.5; the right-hand slope is steeper for the heavier tail.",
    ],
)
def dist_expectile():
    return pl.plot_expectile_curve(_lognormals(), "Do a few large observations carry the mean?")


@card(
    "Correlogram",
    "plot_acf",
    [
        "Stems from zero with a marker at the tip; the band is a ribbon that widens with the lag.",
        "Stems beyond the band are the significant lags (AR(1) decays geometrically).",
    ],
)
def acf_single():
    return pl.plot_acf(_ar1(), "Autocorrelation", lags=30)


@card(
    "Correlogram",
    "plot_acf: signed on ARCH",
    [
        "ARCH series: the signed acf stays inside the band (the sign is unpredictable); the next card shows |x|.",
    ],
)
def acf_arch():
    return pl.plot_acf(_arch(), "acf(x)", "signs look random", lags=30)


@card(
    "Correlogram",
    "plot_acf: absolute",
    [
        "Same series with absolute=True: lags are significant, the ARCH signature.",
    ],
)
def acf_arch_abs():
    return pl.plot_acf(_arch(), "acf(|x|)", "volatility clustering", lags=30, absolute=True)


@card(
    "Correlogram",
    "plot_acf_grid",
    [
        "One panel per series, each with its own band; the AR series stands out, the white noise ones stay inside.",
    ],
)
def acf_grid():
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "AR(1)": _ar1(600).to_numpy(),
            "noise 1": rng.normal(size=600),
            "noise 2": rng.normal(size=600),
            "noise 3": rng.normal(size=600),
        }
    )
    return pl.plot_acf_grid(frame, "Per-series correlograms", lags=25, ncols=2)
