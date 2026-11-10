"""Every plotly plotter takes ``height=`` and ``width=`` and honours them (like plotly.express)."""

import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import plotters as pl

W, H = 480, 333


def _data():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-03", periods=120, freq="B")
    wide = pd.DataFrame(rng.normal(size=(120, 3)).cumsum(0) + 50, index=idx, columns=list("abc"))
    s = wide["a"].rename("s")
    pos = wide.abs() + 1
    long = pd.DataFrame({
        "x": rng.uniform(1, 10, 60), "y": rng.uniform(1, 10, 60), "z": rng.uniform(1, 5, 60),
        "g": np.tile(list("ab"), 30), "p": np.tile(list("uvw"), 20), "m": [f"m{i}" for i in range(60)],
        "lo": np.zeros(60), "hi": np.ones(60) * 2, "pt": rng.uniform(0.2, 1.8, 60),
        "ctx": np.tile(list("xyzw"), 15), "val": rng.uniform(0, 1, 60),
    })  # fmt: skip
    long["xn"] = np.tile([1.0, 2.0, 5.0], 20)
    y = rng.integers(0, 2, 200)
    grid_axis = np.linspace(-2, 2, 21)
    grid = pd.DataFrame(np.add.outer(grid_axis, grid_axis), index=grid_axis, columns=grid_axis)
    counts = {"a": 5, "b": 3, "c": 1}
    quant = pd.DataFrame({0.1: np.full(120, -1.0), 0.5: np.zeros(120), 0.9: np.full(120, 1.0)})
    return locals()


D = _data()
S, WIDE, POS, LONG = D["s"], D["wide"], D["pos"], D["long"]
SQUARE = pd.DataFrame(np.arange(9.0).reshape(3, 3), index=list("abc"), columns=list("xyz"))
HIST, FUT = S.iloc[:90], S.iloc[90:]
RNG = np.random.default_rng(1)
YTRUE = D["y"]
SCORES = pd.Series(YTRUE + RNG.normal(size=200), name="m")
PROBS = pd.Series(RNG.uniform(size=200), name="p")
CALLS = {
    "plot_bars_with_line": lambda w, h: pl.plot_bars_with_line(S, WIDE, width=w, height=h),
    "plot_calendar_heatmap": lambda w, h: pl.plot_calendar_heatmap(S, width=w, height=h),
    "plot_cumulative": lambda w, h: pl.plot_cumulative(WIDE / 100, width=w, height=h),
    "plot_drawdown": lambda w, h: pl.plot_drawdown(WIDE / 100, width=w, height=h),
    "plot_line_band": lambda w, h: pl.plot_line_band(S, S - 1, S + 1, width=w, height=h),
    "plot_rolling": lambda w, h: pl.plot_rolling(S, 5, width=w, height=h),
    "plot_series": lambda w, h: pl.plot_series(WIDE, width=w, height=h),
    "plot_stacked": lambda w, h: pl.plot_stacked(POS, width=w, height=h),
    "plot_series_distribution": lambda w, h: pl.plot_series_distribution(WIDE, width=w, height=h),
    "plot_ccf": lambda w, h: pl.plot_ccf(S, S.shift(1).fillna(0), width=w, height=h),
    "plot_gaps": lambda w, h: pl.plot_gaps(WIDE, width=w, height=h),
    "plot_lag": lambda w, h: pl.plot_lag(S, width=w, height=h),
    "plot_periodogram": lambda w, h: pl.plot_periodogram(S, width=w, height=h),
    "plot_seasonal": lambda w, h: pl.plot_seasonal(S, width=w, height=h),
    "plot_acf": lambda w, h: pl.plot_acf(S, lags=10, width=w, height=h),
    "plot_acf_grid": lambda w, h: pl.plot_acf_grid(WIDE, lags=10, width=w, height=h),
    "plot_pacf": lambda w, h: pl.plot_pacf(S, lags=10, width=w, height=h),
    "plot_calibration": lambda w, h: pl.plot_calibration(PROBS, YTRUE, bins=4, min_count=1, width=w, height=h),
    "plot_forecast": lambda w, h: pl.plot_forecast(HIST, FUT, width=w, height=h),
    "plot_pit": lambda w, h: pl.plot_pit(pd.Series(RNG.uniform(size=100)), width=w, height=h),
    "plot_quantile_calibration": lambda w, h: pl.plot_quantile_calibration(S.reset_index(drop=True), D["quant"], width=w, height=h),
    "plot_residuals": lambda w, h: pl.plot_residuals(S, lags=10, width=w, height=h),
    "plot_box": lambda w, h: pl.plot_box(WIDE, width=w, height=h),
    "plot_cumulative_share": lambda w, h: pl.plot_cumulative_share(POS, width=w, height=h),
    "plot_ecdf": lambda w, h: pl.plot_ecdf(WIDE, width=w, height=h),
    "plot_expectile_curve": lambda w, h: pl.plot_expectile_curve(POS, width=w, height=h),
    "plot_histogram": lambda w, h: pl.plot_histogram(WIDE, width=w, height=h),
    "plot_null_test": lambda w, h: pl.plot_null_test(RNG.normal(size=500), 1.0, width=w, height=h),
    "plot_qq": lambda w, h: pl.plot_qq(S, width=w, height=h),
    "plot_density2d": lambda w, h: pl.plot_density2d(LONG, "x", "y", width=w, height=h),
    "plot_regression": lambda w, h: pl.plot_regression(LONG, "x", "y", width=w, height=h),
    "plot_scatter": lambda w, h: pl.plot_scatter(LONG, "x", "y", width=w, height=h),
    "plot_scatter_matrix": lambda w, h: pl.plot_scatter_matrix(LONG[["x", "y", "z"]], width=w, height=h),
    "plot_shaded_scatter": lambda w, h: pl.plot_shaded_scatter(LONG, "x", "y", "z", width=w, height=h),
    "plot_box_series": lambda w, h: pl.plot_box_series(LONG, "xn", "y", width=w, height=h),
    "plot_bump": lambda w, h: pl.plot_bump(WIDE, width=w, height=h),
    "plot_interval": lambda w, h: pl.plot_interval(LONG.head(8), "lo", "hi", "pt", width=w, height=h),
    "plot_paired_difference": lambda w, h: pl.plot_paired_difference(WIDE, "a", width=w, height=h),
    "plot_rank_heatmap": lambda w, h: pl.plot_rank_heatmap(LONG.drop_duplicates(["m", "ctx"]).head(20), "m", "ctx", "val", width=w, height=h),
    "plot_small_multiples": lambda w, h: pl.plot_small_multiples(LONG, "p", "x", "y", width=w, height=h),
    "plot_tradeoff": lambda w, h: pl.plot_tradeoff(LONG, "x", "y", width=w, height=h),
    "plot_bars": lambda w, h: pl.plot_bars(D["counts"], width=w, height=h),
    "plot_counts": lambda w, h: pl.plot_counts(list("aabbc"), width=w, height=h),
    "plot_dumbbell": lambda w, h: pl.plot_dumbbell(LONG.head(6), "m", "lo", "hi", width=w, height=h),
    "plot_grouped_bars": lambda w, h: pl.plot_grouped_bars(SQUARE, width=w, height=h),
    "plot_lollipop": lambda w, h: pl.plot_lollipop(D["counts"], width=w, height=h),
    "plot_pareto": lambda w, h: pl.plot_pareto(D["counts"], width=w, height=h),
    "plot_waterfall": lambda w, h: pl.plot_waterfall(D["counts"], width=w, height=h),
    "plot_contours": lambda w, h: pl.plot_contours(D["grid"], width=w, height=h),
    "plot_correlation": lambda w, h: pl.plot_correlation(WIDE, width=w, height=h),
    "plot_matrix": lambda w, h: pl.plot_matrix(SQUARE, width=w, height=h),
    "plot_table": lambda w, h: pl.plot_table(SQUARE, width=w, height=h),
    "plot_table_heatmap": lambda w, h: pl.plot_table_heatmap(SQUARE, width=w, height=h),
    "plot_coefficient_path": lambda w, h: pl.plot_coefficient_path(WIDE.set_axis(np.arange(120.0)), width=w, height=h),
    "plot_confusion": lambda w, h: pl.plot_confusion(list("aabbc"), list("abbbc"), width=w, height=h),
    "plot_decision_boundary": lambda w, h: pl.plot_decision_boundary(D["grid"], level=0.0, width=w, height=h),
    "plot_dendrogram": lambda w, h: pl.plot_dendrogram(WIDE.head(10), width=w, height=h),
    "plot_error_curves": lambda w, h: pl.plot_error_curves(WIDE.set_axis(np.arange(120.0)), width=w, height=h),
    "plot_partial_dependence": lambda w, h: pl.plot_partial_dependence({"a": pd.Series([1.0, 2.0], index=[0.0, 1.0])}, width=w, height=h),
    "plot_roc": lambda w, h: pl.plot_roc(YTRUE, SCORES, width=w, height=h),
    "plot_scree": lambda w, h: pl.plot_scree([3, 2, 1], width=w, height=h),
    "plot_grid": lambda w, h: pl.plot_grid({"a": pl.plot_series(S), "b": pl.plot_series(S)}, width=w, height=h),
    "plot_views": lambda w, h: pl.plot_views({"a": pl.plot_series(S), "b": pl.plot_series(S)}, width=w, height=h),
}  # fmt: skip


def _plotly_plotters():
    return [n for n in pl.__all__ if n.startswith("plot_") and inspect.isfunction(getattr(pl, n))]


def test_every_plotly_plotter_declares_width_and_height():
    missing = [
        n
        for n in _plotly_plotters()
        if not {"width", "height"} <= set(inspect.signature(getattr(pl, n)).parameters)
    ]
    assert not missing, f"no width/height on: {missing}"


def test_the_call_table_covers_every_plotter():
    assert sorted(CALLS) == sorted(_plotly_plotters()), "add the new plotter to CALLS"


@pytest.mark.parametrize("name", sorted(CALLS))
def test_width_and_height_reach_the_figure(name):
    fig = CALLS[name](W, H)
    assert isinstance(fig, go.Figure)
    assert fig.layout.width == W, f"{name} ignored width"
    assert fig.layout.height == H, f"{name} ignored height"


@pytest.mark.parametrize("name", sorted(CALLS))
def test_default_width_fills_the_container(name):
    fig = CALLS[name](None, H)
    assert fig.layout.width is None, f"{name} sets a width nobody asked for"
