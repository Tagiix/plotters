"""Gallery cards for the print figures (matplotlib). Rendered as PNG at print size."""

import numpy as np
import pandas as pd

import plotters as pl
from plotters import static as ps

from . import datasets as ds
from .registry import card


@card(
    "Print",
    "plot_lines: legend and dashes",
    [
        "Serif 9 pt text at print size; the lines differ by dash as well as colour (readable when photocopied).",
        "Legend below the axes; subtitle wraps to the axes width instead of running off the figure.",
    ],
)
def st_lines():
    ps.use_print_style()
    return ps.plot_lines(
        ds.levels(4),
        "Model levels",
        "A subtitle that is long enough to need wrapping because the figure is embedded at its own size",
        ylabel="level",
    )


@card(
    "Print",
    "plot_lines: end labels",
    [
        "Names written at the line ends instead of a legend; four curves finishing at the same height get nudged-apart labels.",
        "Percent axis writes a bare % (no backslash).",
    ],
)
def st_label_last():
    ps.use_print_style()
    frame = ds.returns(4).cumsum()
    frame.iloc[-1] = [0.30, 0.302, 0.299, 0.301]
    return ps.plot_lines(
        frame, "Cumulative returns", "labels nudged apart", label_last=True, percent=True
    )


@card(
    "Print",
    "plot_points: Pareto front",
    [
        "Scatter with a dashed Pareto front (lower x and lower y win); one marker shape per group.",
    ],
)
def st_points():
    ps.use_print_style()
    rng = np.random.default_rng(40)
    t = pd.DataFrame({"cost": rng.uniform(1, 10, 25), "group": np.tile(["A", "B", "C"], 9)[:25]})
    t["error"] = 1 / t["cost"] + rng.uniform(0, 0.4, 25)
    return ps.plot_points(t, "cost", "error", "Cost and error", group="group", frontier=True)


@card(
    "Print",
    "plot_ranked_bars",
    [
        "Sorted bars with the value written at each tip: the ranking figure for print.",
    ],
)
def st_ranked():
    ps.use_print_style()
    return ps.plot_ranked_bars(
        {"ewrms": 0.31, "har": 0.27, "garch": 0.35, "range": 0.29, "naive": 0.44},
        "Mean error",
        ascending=True,
    )


@card(
    "Print",
    "plot_heatmap",
    [
        "Diverging matrix with every cell written; the ink is chosen from the measured fill, so the neutral middle stays readable.",
    ],
)
def st_heatmap():
    ps.use_print_style()
    rng = np.random.default_rng(41)
    frame = pd.DataFrame(rng.uniform(-1, 1, (5, 5)), index=list("abcde"), columns=list("vwxyz"))
    return ps.plot_heatmap(frame, "Differences")


@card(
    "Print",
    "plot_function",
    [
        "The shape of two losses, with a dotted vertical reference at zero.",
    ],
)
def st_function():
    ps.use_print_style()
    return ps.plot_function(
        {
            "squared": lambda x: x**2,
            "absolute": np.abs,
            "huber": lambda x: np.where(np.abs(x) < 1, 0.5 * x**2, np.abs(x) - 0.5),
        },
        np.linspace(-3, 3, 200),
        "Loss functions",
        marks=[0.0],
        xlabel="error",
        ylabel="loss",
    )


@card(
    "Print",
    "plot_stems",
    [
        "Correlogram: stems with markers; the band half-width per lag widens, drawn from bartlett_band.",
    ],
)
def st_stems():
    ps.use_print_style()
    rng = np.random.default_rng(42)
    x = np.zeros(500)
    for t in range(1, 500):
        x[t] = 0.5 * x[t - 1] + rng.normal()
    coef = pl.acf_values(x, 20)
    return ps.plot_stems(coef, "Autocorrelation", band=pl.bartlett_band(coef, 500))


@card(
    "Print",
    "plot_panels",
    [
        "A grid with one shared legend; x labels only on the bottom row, y labels only on the left column.",
    ],
)
def st_panels():
    ps.use_print_style()
    lv = ds.levels(2)
    return ps.plot_panels(
        {
            "Panel A": lv,
            "Panel B": lv * 0.9,
            "Panel C": lv.iloc[::-1].set_axis(lv.index),
            "Panel D": lv + 5,
        },
        "Four panels",
        ncols=2,
        xlabel="date",
        ylabel="level",
    )
