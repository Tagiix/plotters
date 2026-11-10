"""Gallery cards for the categorical and relationship charts."""

import numpy as np
import pandas as pd

import plotters as pl

from . import datasets as ds
from .registry import card


@card(
    "Categorical",
    "plot_bars: dict of counts",
    [
        "Straight from a {label: count} dict: largest first at the top, values written at the tips, one hue.",
    ],
)
def cat_bars():
    return pl.plot_bars(
        {"cats": 42, "dogs": 31, "birds": 12, "fish": 9, "rabbits": 4},
        "Pets owned",
        "plot_bars(dict)",
    )


@card(
    "Categorical",
    "plot_bars: shares",
    [
        "normalize=True: shares in percent instead of counts (the honest replacement for a pie chart).",
    ],
)
def cat_bars_share():
    return pl.plot_bars(
        {"Q1": 120, "Q2": 90, "Q3": 60, "Q4": 30}, "Share of sales", normalize=True, sort=False
    )


@card(
    "Categorical",
    "plot_counts",
    [
        "Raw observations counted and drawn in one call; top=6 keeps the six most frequent values.",
    ],
)
def cat_counts():
    rng = np.random.default_rng(30)
    words = rng.choice(
        [f"word{i}" for i in range(12)],
        400,
        p=np.r_[np.linspace(0.2, 0.01, 12)] / np.linspace(0.2, 0.01, 12).sum(),
    )
    return pl.plot_counts(words, "Most frequent values", top=6)


@card(
    "Categorical",
    "plot_grouped_bars",
    [
        "Categories on x, one coloured bar per series; the legend sits below the plot.",
    ],
)
def cat_grouped():
    frame = pd.DataFrame(
        {"2022": [4.1, 3.2, 5.0], "2023": [4.6, 2.9, 5.5], "2024": [5.2, 3.8, 5.1]},
        index=["North", "South", "West"],
    )
    return pl.plot_grouped_bars(frame, "Revenue by region", ylabel="revenue")


@card(
    "Categorical",
    "plot_grouped_bars: stacked shares",
    [
        "normalize=True stacks to 100% so each category shows its composition.",
    ],
)
def cat_stacked():
    frame = pd.DataFrame({"A": [4, 3, 5], "B": [2, 5, 1], "C": [1, 1, 3]}, index=["u", "v", "w"])
    return pl.plot_grouped_bars(frame, "Composition", normalize=True)


@card(
    "Categorical",
    "plot_lollipop",
    [
        "Stems from zero with dots and written values; negative values extend to the left.",
    ],
)
def cat_lollipop():
    return pl.plot_lollipop(
        {"alpha": 2.4, "beta": 1.1, "gamma": -0.6, "delta": -1.9, "eps": 0.3},
        "Contribution by factor",
    )


@card(
    "Categorical",
    "plot_dumbbell",
    [
        "Before (muted) and after (accent) per item joined by a line; largest moves at the top.",
    ],
)
def cat_dumbbell():
    frame = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E"],
            "before": [0.62, 0.55, 0.71, 0.48, 0.66],
            "after": [0.58, 0.41, 0.69, 0.52, 0.47],
        }
    )
    return pl.plot_dumbbell(frame, "model", "before", "after", "Error before and after the change")


@card(
    "Categorical",
    "plot_waterfall",
    [
        "Contributions in the order given, increases and decreases in two hues, a grey closing total.",
    ],
)
def cat_waterfall():
    return pl.plot_waterfall(
        {"Start": 100, "Price": 12, "Volume": 7, "Costs": -18, "FX": -5},
        "Bridge from start to total",
    )


@card(
    "Time series",
    "plot_line_band",
    [
        "A line with a translucent ribbon in the same colour; two series have independent ribbons.",
    ],
)
def cat_band():
    lv = ds.levels(2)
    spread = pd.Series(np.linspace(1, 6, len(lv)), index=lv.index)
    return pl.plot_line_band(
        lv,
        {c: lv[c] - spread for c in lv},
        {c: lv[c] + spread for c in lv},
        "Forecast with uncertainty",
    )


@card(
    "Relationships",
    "plot_regression",
    [
        "Scatter with an OLS line and a 95% band for the mean; the subtitle reports slope and R-squared.",
    ],
)
def cat_regression():
    rng = np.random.default_rng(31)
    x = rng.normal(size=150)
    return pl.plot_regression(
        pd.DataFrame({"x": x, "y": 0.8 * x + rng.normal(0, 0.6, 150)}), "x", "y", "Regression"
    )


@card(
    "Relationships",
    "plot_scatter_matrix",
    [
        "Histograms on the diagonal, pairwise scatter below it; axis titles on the outer edge only.",
    ],
)
def cat_matrix():
    rng = np.random.default_rng(32)
    a = rng.normal(size=200)
    return pl.plot_scatter_matrix(
        pd.DataFrame({"a": a, "b": 0.7 * a + rng.normal(0, 0.5, 200), "c": rng.normal(size=200)}),
        "Pairwise",
    )


@card(
    "Time series",
    "plot_calendar_heatmap",
    [
        "Weekday rows by week columns, shaded by value; month ticks along the bottom.",
    ],
)
def cat_calendar():
    idx = pd.date_range("2023-01-01", "2023-12-31")
    rng = np.random.default_rng(33)
    return pl.plot_calendar_heatmap(
        pd.Series(
            rng.gamma(2, 1, len(idx)) * (1 + 0.5 * np.sin(np.arange(len(idx)) / 30)), index=idx
        ),
        "Daily activity",
    )
