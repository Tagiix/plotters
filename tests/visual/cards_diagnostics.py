"""Gallery cards for time-series diagnostics, bump chart, table and range buttons."""

import numpy as np
import pandas as pd

import plotters as pl

from . import datasets as ds
from .registry import card


def _monthly(years=8, seed=80):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=12 * years, freq="MS")
    t = np.arange(len(idx))
    return pd.Series(
        10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.6, len(t)),
        index=idx,
        name="sales",
    )


@card(
    "Diagnostics",
    "plot_gaps",
    [
        "One row per series, time along x, coloured by the share of each stretch that is missing; row labels carry the overall share and are not clipped.",
        "A long outage (series B) looks different from scattered single gaps (series C).",
    ],
)
def dg_gaps():
    rng = np.random.default_rng(81)
    idx = pd.date_range("2022-01-03", periods=500, freq="B")
    a = pd.Series(rng.normal(size=500), index=idx, name="A: complete")
    b = a.copy().rename("B: outage")
    b.iloc[150:260] = np.nan
    c = a.copy().rename("C: scattered")
    c[rng.uniform(size=500) < 0.12] = np.nan
    d = a.copy().rename("D: starts late").iloc[300:]
    return pl.plot_gaps({"A": a, "B": b, "C": c, "D": d}, "Data availability", sort=True)


@card(
    "Diagnostics",
    "plot_seasonal",
    [
        "One line per year laid over Jan to Dec: older years fainter, recent ones stronger; the dashed line is the average.",
        "Parallel lines stacked above one another show the upward trend; the shared shape is the seasonality.",
    ],
)
def dg_seasonal():
    return pl.plot_seasonal(_monthly(), "Seasonal plot", ylabel="sales")


@card(
    "Diagnostics",
    "plot_seasonal: weekday",
    [
        "One line per ISO week across Monday to Sunday: the weekday pattern of an intraday-like series.",
    ],
)
def dg_weekday():
    rng = np.random.default_rng(82)
    idx = pd.date_range("2023-01-02", periods=7 * 10, freq="D")
    s = pd.Series(
        100 + 20 * np.sin(np.arange(len(idx)) * 2 * np.pi / 7) + rng.normal(0, 4, len(idx)),
        index=idx,
    )
    return pl.plot_seasonal(s, "Weekly pattern", period="weekday", max_cycles=10)


@card(
    "Diagnostics",
    "plot_lag",
    [
        "Each value against the one k steps earlier; the correlation is in each panel title. Dotted diagonal is y = x.",
        "Persistence (lag 1) shows as a tight cloud along the diagonal; it loosens at longer lags.",
    ],
)
def dg_lag():
    rng = np.random.default_rng(83)
    x = np.zeros(600)
    for t in range(1, 600):
        x[t] = 0.85 * x[t - 1] + rng.normal()
    return pl.plot_lag(pd.Series(x), "Does today resemble yesterday?", lags=[1, 3, 10])


@card(
    "Diagnostics",
    "plot_ccf",
    [
        "Cross-correlation in both directions; the peak at a negative lag means the second series follows the first.",
        "Flat band is what unrelated white noise allows.",
    ],
)
def dg_ccf():
    rng = np.random.default_rng(84)
    rain = pd.Series(rng.normal(size=800), name="rain")
    river = pd.Series(np.roll(rain.to_numpy(), 4) * 0.8 + rng.normal(0, 0.5, 800), name="river")
    return pl.plot_ccf(rain, river, "Does rain lead the river?", lags=15)


@card(
    "Diagnostics",
    "plot_periodogram",
    [
        "Variance by frequency: two spikes labelled with their periods (12 and 30). Log power axis.",
    ],
)
def dg_periodogram():
    rng = np.random.default_rng(85)
    t = np.arange(480)
    s = np.sin(2 * np.pi * t / 12) + 0.7 * np.sin(2 * np.pi * t / 30) + rng.normal(0, 0.5, 480)
    return pl.plot_periodogram(s, "Spectrum", peaks=2)


@card(
    "Diagnostics",
    "plot_periodogram: period axis",
    [
        "Same idea in the series' own units: monthly data in years (spacing=1/12), x as period on a log axis; the annual spike sits at 1 year.",
    ],
)
def dg_periodogram_period():
    s = _monthly(years=20)
    return pl.plot_periodogram(s, "Spectrum by period", x="period", spacing=1 / 12, peaks=2)


@card(
    "Comparisons",
    "plot_bump",
    [
        "Rank of each model at every step (1 at the top); the highlighted models are in colour, the rest thin and grey.",
        "Names are written at the right end.",
    ],
)
def dg_bump():
    rng = np.random.default_rng(86)
    base = np.linspace(0, 1, 12)[:, None] * np.array([0.2, -0.3, 0.1, 0.0, -0.1, 0.25])
    frame = pd.DataFrame(base + rng.normal(0, 0.12, (12, 6)), index=pd.date_range("2023-01-01", periods=12, freq="MS"),
                         columns=["ewrms", "har", "garch", "range", "naive", "ensemble"])  # fmt: skip
    return pl.plot_bump(frame, "Model ranking over time", highlight=["har", "ensemble"])


@card(
    "Matrices & tables",
    "plot_table",
    [
        "A DataFrame as a plain styled table: numbers right-aligned and formatted per column, missing shown as a dash, rows alternating.",
    ],
)
def tb_table():
    t = pd.DataFrame(
        {
            "model": ["ewrms", "har", "garch", "naive"],
            "mae": [0.31, 0.27, 0.35, np.nan],
            "share": [0.31, 0.42, 0.18, 0.09],
            "n": [1200, 1200, 1180, 1200],
        }
    )
    return pl.plot_table(
        t, "Model summary", column_formats={"share": ".0%", "n": ",.0f"}, index=False
    )


@card(
    "Comparisons",
    "group_summary + plot_interval",
    [
        "group_summary computes the mean and a 95% interval per group; plot_interval draws them. Hollow = the interval crosses 0.",
    ],
)
def tb_group_summary():
    rng = np.random.default_rng(87)
    frame = pd.DataFrame(
        {
            "arm": np.repeat(["control", "A", "B", "C"], 60),
            "effect": np.r_[
                rng.normal(0, 1, 60),
                rng.normal(0.4, 1, 60),
                rng.normal(0.9, 1, 60),
                rng.normal(0.1, 1.4, 60),
            ],
        }
    )
    summary = pl.group_summary(frame, "arm", "effect").rename_axis("arm").reset_index()
    return pl.plot_interval(
        summary, "low", "high", "point", "Effect by arm", label="arm", reference=0.0
    )


@card(
    "Time series",
    "plot_series: range buttons",
    [
        "Quick date-range buttons (1m, 3m, 6m, YTD, 1y, all) at the top right; click one and the x range jumps.",
    ],
)
def ts_range_buttons():
    return pl.plot_series(ds.levels(2, n=600), "Levels with range buttons", range_buttons=True)
