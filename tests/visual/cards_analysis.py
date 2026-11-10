"""Gallery cards for statistics, analysis and views."""

import plotters as pl

from . import datasets as ds
from .registry import card


@card(
    "Analysis",
    "plot_series_distribution",
    [
        "Histogram with a dropdown: pooled first, then one view per series.",
        "Dotted vertical markers at 1%, 50%, 99% with labels; they move when the dropdown changes.",
        "Statistics block at the top right changes with the view. Dropdown sits below the subtitle.",
    ],
)
def an_distribution():
    return pl.plot_series_distribution(
        ds.returns(3), "Return distribution", "Fat tails", pooled=True, bins=60
    )


@card(
    "Analysis",
    "Distribution with normal reference",
    [
        "Dashed N(0, 1) overlay on a standardised series; the tails sit above it (excess kurtosis).",
    ],
)
def an_distribution_normal():
    z = pl.normalize_series(ds.returns(1), "std").series
    return pl.plot_series_distribution(
        z, "Standardised returns", "vs N(0, 1), not a fitted normal", normal_reference=True, bins=60
    )


@card(
    "Analysis",
    "Histogram window keeps tail mass",
    [
        "Window clipped to [-0.02, 0.02]; the bars are lower than the unclipped version because the density denominator still counts the outside values.",
    ],
)
def an_distribution_window():
    return pl.plot_series_distribution(
        ds.returns(1), "hist_range keeps the denominator", hist_range=(-0.02, 0.02), bins=40
    )


@card(
    "Analysis",
    "analyze_series: values",
    [
        "Dropdown: pooled, then each series. Turnover ratio annotation at the top right changes with the view.",
    ],
)
def an_values():
    return pl.analyze_series(ds.returns(3), "Returns")["figures"]["values"]


@card(
    "Analysis",
    "analyze_series: cumulative absolute",
    [
        "Cumulative sum of magnitudes: monotonically rising. Pooled averages magnitudes before accumulating.",
    ],
)
def an_cum_abs():
    return pl.analyze_series(ds.returns(3), "Returns")["figures"]["cumulative_absolute"]


@card(
    "Analysis",
    "analyze_series: normalised histogram",
    [
        "With normalize='std' every series has unit spread, so the histograms share a scale.",
    ],
)
def an_norm_hist():
    return pl.analyze_series(ds.returns(3), "Returns", normalize="std")["figures"]["histogram"]


@card(
    "Analysis",
    "compare_series: paths",
    [
        "Cumulative paths of all inputs on the same dates.",
    ],
)
def an_compare_paths():
    return pl.compare_series(ds.returns(3), "Models", reference="alpha")["figures"]["cumulative"]


@card(
    "Analysis",
    "compare_series: difference",
    [
        "Cumulative difference (reference minus others): steady slope or one cliff?",
    ],
)
def an_compare_diff():
    return pl.compare_series(ds.returns(3), "Models", reference="alpha")["figures"]["difference"]


@card(
    "Views",
    "plot_views: different axes",
    [
        "Two prepared figures with different axis titles and a log y axis in the second; switching changes them together.",
        "Annotation at the top right changes with the dropdown.",
    ],
)
def views_axes():
    g = ds.positive_growth()
    raw = pl.plot_series(g, "raw", ylabel="level", bands={"target": 3.0})
    logged = pl.plot_series(g, "log", logy=True, ylabel="level (log)")
    fig = pl.plot_views(
        {"Raw": raw, "Log scale": logged},
        "Same series, two scales",
        "Dropdown switches axes together",
    )
    return pl.annotate_views(fig, {"Raw": "linear axis", "Log scale": "log axis"})
