"""Gallery cards for the theme, layout and time-series modules."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import plotters as pl

from . import datasets as ds
from .registry import card

# ----------------------------------------------------------------------------- layout


@card(
    "Layout",
    "Three series",
    [
        "Title is left-aligned with the plot edge; subtitle is smaller and muted.",
        "Legend sits below the plot, left-aligned, never on top of the title.",
        "Three lines in blue, rust, cyan; all solid.",
    ],
)
def layout_three():
    return pl.plot_series(ds.levels(3), "Model levels", "Random walks, seeded", ylabel="level")


@card(
    "Layout",
    "Single series",
    [
        "No legend: the title names the one series.",
        "Bottom margin shrinks accordingly.",
    ],
)
def layout_single():
    return pl.plot_series(ds.levels(1)["model_0"], "One series", "A single line needs no legend")


@card(
    "Layout",
    "Short figure (300 px)",
    [
        "Legend distance below the axis matches the 420 px card exactly (pixel offset, not fraction).",
    ],
)
def layout_short():
    return pl.plot_series(ds.levels(3), "Height 300", "Legend offset is in pixels", height=300)


@card(
    "Layout",
    "Tall figure (800 px)",
    [
        "Legend still sits the same few pixels under the axis, not drifting down.",
    ],
)
def layout_tall():
    return pl.plot_series(ds.levels(3), "Height 800", "Legend offset is in pixels", height=800)


@card(
    "Layout",
    "Long title and subtitle",
    [
        "Long title stays left-aligned and does not collide with the legend or the plot.",
    ],
)
def layout_long():
    return pl.plot_series(
        ds.levels(3),
        "A rather long title that describes exactly what is being shown in this figure",
        "and a subtitle that goes on for a while to check how the header behaves with lots of text in it",
    )


@card(
    "Layout",
    "Many series (30)",
    [
        "Past 12 entries the legend becomes a vertical column on the right and scrolls; the plot keeps its height.",
        "Only the first 20 series start drawn; the rest are greyed out in the legend.",
        "Dashes change after the 8th series so a repeated colour is still a different line.",
    ],
)
def layout_many():
    return pl.plot_series(ds.levels(30), "Thirty series", "20 drawn on open, 10 in the legend")


@card(
    "Layout",
    "Log axis crossing 1",
    [
        "Y ticks read 0.2 0.5 1 2 5 ... in full, never '2 3 4 5 ... 1 2 3'.",
        "Dotted target line sits at exactly 3 and the event line at the marked date, with labels.",
    ],
)
def layout_log():
    g = ds.positive_growth()
    return pl.plot_series(
        g,
        "Log axis",
        "Data crosses 1",
        logy=True,
        bands={"target": 3.0},
        markers={"event": g.index[150]},
    )


@card(
    "Layout",
    "Log x axis",
    [
        "Both axes logarithmic with readable ticks; the vertical marker sits at x = 100 with its label.",
    ],
)
def layout_logx():
    x = np.logspace(0, 3, 150)
    s = pd.Series(x**0.8, index=x, name="power law")
    return pl.plot_series(
        s, "Log-log", "Straight line on log-log", logx=True, logy=True, markers={"m": 100.0}
    )


@card(
    "Layout",
    "Plain plotly figure",
    [
        "A bare go.Figure created after set_style() picks up fonts, colours, grid and background: no finish() call.",
    ],
)
def layout_bare():
    fig = go.Figure()
    for i in range(3):
        fig.add_scatter(y=np.cumsum(np.random.default_rng(i).normal(size=60)), name=f"s{i}")
    fig.update_layout(title="Bare go.Figure", height=380)
    return fig


# ----------------------------------------------------------------------------- time series


@card(
    "Time series",
    "plot_series: percent",
    [
        "Y axis ticks read as percentages (0.01 -> 1%); hover shows two decimals.",
    ],
)
def ts_percent():
    return pl.plot_series(ds.returns(3), "Daily returns", "Fractions formatted as %", percent=True)


@card(
    "Time series",
    "plot_series: slider and fill",
    [
        "Range slider under the axis; legend sits below the slider, not on top of it.",
        "Each line is filled down to zero with a light tint.",
    ],
)
def ts_slider():
    return pl.plot_series(
        ds.levels(2), "Slider and fill", "Drag the slider", slider=True, fill=True
    )


@card(
    "Time series",
    "plot_series: views dropdown",
    [
        "Dropdown at the top left, below the title and above the plot.",
        "Switching changes the visible lines; 'pooled' (named in no view) stays in every view.",
        "The legend lists only the lines of the current view.",
    ],
)
def ts_views():
    lv = ds.levels(6)
    lv["pooled"] = lv.mean(axis=1)
    return pl.plot_series(
        lv,
        "Views",
        "pooled stays visible in all views",
        views={
            "first pair": ["model_0", "model_1"],
            "second pair": ["model_2", "model_3"],
            "third pair": ["model_4", "model_5"],
        },
    )


@card(
    "Time series",
    "plot_series: max_visible",
    [
        "max_visible=8: eight lines start drawn; the other 12 are greyed out in the legend, one click to add.",
    ],
)
def ts_maxvisible():
    return pl.plot_series(
        ds.levels(20), "max_visible=8", "Click a greyed legend entry to add it", max_visible=8
    )


@card(
    "Time series",
    "plot_series: NaN gaps",
    [
        "Lines break at missing values instead of bridging them.",
    ],
)
def ts_nan():
    lv = ds.levels(2)
    lv.iloc[100:130, 0] = np.nan
    return pl.plot_series(lv, "Gaps", "Missing values break the line")


@card(
    "Time series",
    "plot_series: series= order",
    [
        "Legend order and colours follow the series= list (gamma first = blue), not the frame's order.",
    ],
)
def ts_order():
    return pl.plot_series(
        ds.returns(4), "series= fixes order", "gamma, alpha", series=["gamma", "alpha"]
    )


@card(
    "Time series",
    "plot_bars_with_line",
    [
        "Realised values as muted bars, predictions as coloured lines, one shared axis.",
    ],
)
def ts_bars_line():
    r = ds.returns(3).iloc[:60]
    return pl.plot_bars_with_line(
        r["alpha"],
        {"model A": r["alpha"].rolling(3).mean(), "model B": r["alpha"] * 0.5},
        "Did it track or lag?",
        "Bars = realised",
        percent=True,
    )


@card(
    "Time series",
    "plot_cumulative",
    [
        "Starts at 1.0. Compounded curve ends above the summed curve for positive drift.",
    ],
)
def ts_cumulative():
    return pl.plot_cumulative(ds.returns(3) + 0.001, "Growth of 1", "compound=True", compound=True)


@card(
    "Time series",
    "plot_drawdown",
    [
        "Filled areas below zero: the area is the quantity. Axis in percent, never above 0.",
    ],
)
def ts_drawdown():
    return pl.plot_drawdown(ds.returns(2), "Drawdown", "Drop from the running peak")


@card(
    "Time series",
    "plot_stacked",
    [
        "Parts of a whole, stacked, one hue each.",
    ],
)
def ts_stacked():
    return pl.plot_stacked(ds.shares(4), "Composition", "Stacked parts")


@card(
    "Time series",
    "plot_stacked: normalised",
    [
        "Shares sum to 100% at every date; axis shows a % suffix.",
    ],
)
def ts_stacked_norm():
    return pl.plot_stacked(ds.shares(4), "Composition as shares", "normalize=True", normalize=True)


@card(
    "Time series",
    "plot_rolling",
    [
        "Raw series in light ink behind the bold rolling statistic of the same colour.",
        "Legend has one entry per series, not two.",
    ],
)
def ts_rolling():
    return pl.plot_rolling(ds.returns(2), 30, "Rolling mean", how="mean")
