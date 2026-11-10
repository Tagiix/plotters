"""Gallery cards for the matrix charts."""

import numpy as np
import pandas as pd

import plotters as pl

from .registry import card


def _data(n=300) -> pd.DataFrame:
    rng = np.random.default_rng(12)
    base = rng.standard_t(5, n)
    return pd.DataFrame(
        {
            "alpha": base,
            "beta": 0.9 * base + 0.3 * rng.standard_t(5, n),
            "gamma": -0.5 * base + rng.standard_t(5, n),
            "delta": rng.standard_t(5, n),
            "eps": 0.3 * base + rng.standard_t(5, n),
        }
    )


@card(
    "Matrices",
    "plot_correlation",
    [
        "Diverging: one hue per pole with a neutral midpoint (grey, dark in dark mode); diagonal blank.",
        "Cell numbers stay readable on every fill, including the dark midpoint (ink chosen from measured luminance).",
    ],
)
def mat_corr():
    return pl.plot_correlation(_data(), "Correlation", "pearson, diagonal masked")


@card(
    "Matrices",
    "plot_correlation: spearman, fixed range",
    [
        "Same data with rank correlation on a fixed -1..1 scale, comparable between charts.",
    ],
)
def mat_corr_spearman():
    return pl.plot_correlation(
        _data(), "Spearman", "fixed_range=True", method="spearman", fixed_range=True
    )


@card(
    "Matrices",
    "plot_matrix: non-straddling",
    [
        "All values positive: single-hue sequential ramp, pale to deep, with cell values written in.",
    ],
)
def mat_seq():
    rng = np.random.default_rng(13)
    frame = pd.DataFrame(
        rng.uniform(0.1, 3.0, (6, 5)),
        index=[f"item {i}" for i in range(6)],
        columns=[f"ctx {c}" for c in "ABCDE"],
    )
    return pl.plot_matrix(frame, "Errors by item and context", "sequential, range from the data")


@card(
    "Matrices",
    "plot_matrix: symmetric about zero",
    [
        "Values straddle zero: diverging ramp with the range forced symmetric, so grey really is zero.",
    ],
)
def mat_div():
    rng = np.random.default_rng(14)
    frame = pd.DataFrame(
        rng.normal(0, 1, (6, 5)) + 0.5,
        columns=[f"ctx {c}" for c in "ABCDE"],
        index=[f"item {i}" for i in range(6)],
    )
    return pl.plot_matrix(frame, "Differences", "range symmetric about 0 although the data is not")


@card(
    "Matrices",
    "plot_table_heatmap",
    [
        "Columns on different scales: n, mean, std, skewness each shaded within its own column.",
        "Actual values are written in the cells; the constant column n is not shaded.",
    ],
)
def mat_table():
    stats = pl.series_statistics(_data())[["n", "mean", "std", "skewness", "excess_kurtosis"]]
    return pl.plot_table_heatmap(stats, "Moments table", "each column shaded on its own range")


@card(
    "Matrices",
    "plot_contours",
    [
        "Labelled level curves with exact values; dashed y = x where the two arguments agree (f vanishes there).",
    ],
)
def mat_contours():
    g = np.linspace(0.1, 3.0, 80)
    a, b = np.meshgrid(g, g)
    frame = pd.DataFrame(np.log(a / b) ** 2, index=g, columns=g)
    return pl.plot_contours(
        frame,
        "Loss f(a, b) = log(a / b)^2",
        "zero along the diagonal",
        diagonal=True,
        xlabel="a",
        ylabel="b",
        height=520,
    )


@card(
    "Matrices",
    "plot_contours: log axes",
    [
        "Both axes logarithmic with readable ticks; contours stay aligned with the grid.",
    ],
)
def mat_contours_log():
    g = np.logspace(-1, 1, 70)
    a, b = np.meshgrid(g, g)
    frame = pd.DataFrame(np.abs(a - b) / (a + b), index=g, columns=g)
    return pl.plot_contours(
        frame, "Relative gap", "log-log", logx=True, logy=True, diagonal=True, height=520
    )
