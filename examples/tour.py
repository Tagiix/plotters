"""A tour of the everyday charts, one file each. Run: python examples/tour.py (writes tour_*.html)."""

import numpy as np
import pandas as pd

import plotters as pl

rng = np.random.default_rng(1)
pl.set_style(dark=False)

# Counts straight from a dict, with error bars from group_summary.
pl.save_html(pl.plot_bars({"cats": 42, "dogs": 31, "birds": 12}, "Pets owned"), "tour_bars.html")

trial = pd.DataFrame({"arm": np.repeat(["control", "A", "B"], 80),
                      "effect": np.r_[rng.normal(0, 1, 80), rng.normal(0.5, 1, 80), rng.normal(1.0, 1, 80)]})  # fmt: skip
summary = pl.group_summary(trial, "arm", "effect").rename_axis("arm").reset_index()
pl.save_html(
    pl.plot_interval(summary, "low", "high", "point", "Effect by arm", label="arm", reference=0.0),
    "tour_interval.html",
)

# A bubble chart: x, y and size are column names of a long frame.
cities = pd.DataFrame(
    {"gdp": rng.uniform(1, 60, 12), "life": rng.uniform(55, 85, 12), "pop": rng.uniform(1, 300, 12)}
)
pl.save_html(
    pl.plot_scatter(cities, "gdp", "life", "Wealth and life expectancy", size="pop"),
    "tour_bubbles.html",
)

# Diagnostics on a monthly series: seasonality and cycles.
idx = pd.date_range("2016-01-01", periods=96, freq="MS")
t = np.arange(96)
sales = pd.Series(
    10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.6, 96), index=idx, name="sales"
)
pl.save_html(pl.plot_seasonal(sales, "Seasonal plot"), "tour_seasonal.html")
pl.save_html(
    pl.plot_periodogram(sales, "Spectrum", spacing=1 / 12, x="period"), "tour_spectrum.html"
)

# Several finished figures in one grid.
pl.save_html(
    pl.plot_grid(
        {"Level": pl.plot_series(sales, "Level"), "Seasonal": pl.plot_seasonal(sales, "Seasonal")}
    ),
    "tour_grid.html",
)
