"""Gallery cards for the small, common tasks."""

import numpy as np
import pandas as pd

import plotters as pl

from . import datasets as ds
from .registry import card


@card(
    "Time series",
    "plot_series: shaded spans",
    [
        "Two shaded bands behind the lines with their labels at the top; the lines stay readable on top.",
    ],
)
def sm_spans():
    lv = ds.levels(2)
    idx = lv.index
    return pl.plot_series(
        lv,
        "Levels with events",
        spans={"outage": (idx[80], idx[110]), "regime change": (idx[250], idx[300])},
    )


@card(
    "Time series",
    "plot_series: step",
    [
        "A step function: each value holds until the next observation, with right-angled corners.",
    ],
)
def sm_step():
    s = pd.Series(
        [1.0, 1.0, 2.5, 2.5, 2.5, 1.8, 3.0, 3.0, 2.2],
        index=pd.date_range("2023-01-01", periods=9, freq="MS"),
        name="policy rate",
    )
    return pl.plot_series(s, "Rate held between decisions", step=True, ylabel="%")


@card(
    "Relationships",
    "plot_scatter: bubbles",
    [
        "Marker area (not diameter) is proportional to the third column; the subtitle says so.",
        "Two groups in two hues and shapes; labelled bubbles stay readable.",
    ],
)
def sm_bubbles():
    rng = np.random.default_rng(70)
    frame = pd.DataFrame(
        {
            "gdp": rng.uniform(1, 60, 14),
            "life": rng.uniform(55, 85, 14),
            "pop": rng.uniform(1, 300, 14),
            "region": np.tile(["north", "south"], 7),
        }
    )
    frame["name"] = [f"c{i}" for i in range(14)]
    return pl.plot_scatter(
        frame, "gdp", "life", "Wealth and life expectancy", size="pop", group="region", label="name"
    )


@card(
    "Relationships",
    "plot_shaded_scatter: four variables",
    [
        "x, y, colour (diverging about 1, log ratio) and size on one chart.",
    ],
)
def sm_shaded_bubbles():
    rng = np.random.default_rng(71)
    t = pd.DataFrame(
        {
            "x": rng.uniform(1, 80, 40),
            "y": rng.uniform(0.1, 1, 40),
            "n": rng.uniform(5, 100, 40),
            "ratio": np.exp(rng.normal(0, 0.7, 40)),
        }
    )
    return pl.plot_shaded_scatter(
        t, "x", "y", "ratio", "Four variables in one chart", size="n", log_shade=True, reference=1.0
    )


@card(
    "Categorical",
    "plot_bars: error bars",
    [
        "Means with asymmetric confidence intervals as whiskers; the values are written inside the bars so they never collide.",
    ],
)
def sm_bar_errors():
    means = {"control": 4.2, "drug A": 5.6, "drug B": 6.1, "drug C": 4.9}
    minus = {"control": 0.5, "drug A": 0.7, "drug B": 0.4, "drug C": 0.9}
    plus = {"control": 0.6, "drug A": 0.9, "drug B": 0.5, "drug C": 1.1}
    return pl.plot_bars(
        means, "Mean response", "bars: mean, whiskers: 95% interval", error=(minus, plus), fmt=".1f"
    )


@card(
    "Categorical",
    "plot_grouped_bars: error bars",
    [
        "Grouped bars each with their own whisker, in matching positions.",
    ],
)
def sm_grouped_errors():
    f = pd.DataFrame(
        {"2022": [4.1, 3.2, 5.0], "2023": [4.6, 2.9, 5.5]}, index=["North", "South", "West"]
    )
    return pl.plot_grouped_bars(f, "Revenue by region", error=f * 0.08, ylabel="revenue")


@card(
    "Categorical",
    "plot_pareto",
    [
        "Causes ranked by share with the running total; the dotted 80% line shows how few causes make up most of it.",
        "The tail is folded into one 'Other' bar.",
    ],
)
def sm_pareto():
    defects = {
        "scratch": 120,
        "dent": 85,
        "misaligned": 60,
        "paint": 30,
        "crack": 18,
        "burr": 12,
        "stain": 9,
        "other": 6,
    }
    return pl.plot_pareto(defects, "Defects by type", top=5)


@card(
    "Views",
    "plot_grid",
    [
        "Three finished figures of different kinds placed in one grid, each keeping its own axes, titles and reference lines.",
        "The shared series name is listed in the legend once.",
    ],
)
def sm_grid():
    lv = ds.levels(1)["model_0"].rename("model")
    a = pl.plot_series(lv, "Level", ylabel="level", bands={"start": float(lv.iloc[0])})
    b = pl.plot_rolling(lv.diff().dropna(), 30, "Rolling mean of changes", how="mean")
    c = pl.plot_bars(
        {"up": int((lv.diff() > 0).sum()), "down": int((lv.diff() < 0).sum())}, "Days up and down"
    )
    d = pl.plot_ecdf(lv.diff().dropna().rename("model"), "ECDF of changes")
    return pl.plot_grid([a, b, c, d], "Four views of one series", ncols=2)
