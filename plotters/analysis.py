"""Series analysis: statistics and figures together. Convention A throughout.

Use the individual functions in :mod:`plotters.stats` for one table, or
:func:`analyze_series` for the table and the paths in one call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import _color
from . import style as _style
from ._types import SeriesLike
from .frames import as_series_dict, pick
from .layout import HEIGHT
from .stats import (
    _finite,
    normalize_series,
    series_statistics,
    series_turnover_ratio,
)
from .timeseries import plot_series
from .views import annotate_views, plot_views

__all__ = ["analyze_series", "compare_series", "plot_series_distribution"]

_DEFAULT_DISPLAY = ("mean", "std", "skewness", "excess_kurtosis")
_POOLED = "pooled"


def _pooled_statistics(
    frames: Mapping[str, pd.Series],
    quantiles: Sequence[float],
    thresholds: Sequence[float] | None,
    metrics: Mapping[str, Callable[[pd.Series], float]] | None,
    per_series: pd.DataFrame,
    tail: Mapping[str, Any],
    turnover: Callable[[pd.Series], float] | None,
) -> pd.Series:
    """Statistics of every observation stacked with equal weight per observation.

    Turnover keeps differences within each input series. A custom metric is pooled as the
    observation-count weighted mean of the per-series values, and so is a custom ``turnover``.
    Autocorrelation and rolling columns are per series only, so the pooled row leaves them NaN.
    """
    stacked = pd.concat(
        [_finite(s).reset_index(drop=True) for s in frames.values()], ignore_index=True
    )
    row = series_statistics(
        stacked.rename(_POOLED), quantiles=quantiles, thresholds=thresholds, **tail
    ).iloc[0]
    diffs = [
        pd.to_numeric(s, errors="coerce").astype(float).where(lambda v: np.isfinite(v)).diff().abs()
        for s in frames.values()
    ]
    levels = [_finite(s).abs() for s in frames.values()]
    all_diff = pd.concat(diffs, ignore_index=True).mean()
    all_level = pd.concat(levels, ignore_index=True).mean()
    row["turnover_ratio"] = float(all_diff / all_level) if all_level else np.nan
    if turnover is not None:
        row["turnover_ratio"] = float(
            np.average(per_series["turnover_ratio"], weights=per_series["n"])
        )
    weights = per_series["n"]
    for name in metrics or {}:
        row[name] = float(np.average(per_series[name], weights=weights))
    return row


def plot_series_distribution(
    data: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    series: Sequence[str] | None = None,
    bins: int = 50,
    pooled: bool = False,
    statistics: bool = True,
    display_stats: Sequence[str] = _DEFAULT_DISPLAY,
    quantiles: Sequence[float] | None = (0.01, 0.5, 0.99),
    normal_reference: bool = False,
    hist_range: tuple[float, float] | None = None,
    height: int = HEIGHT,
    width: int | None = None,
) -> go.Figure:
    """Histogram with a dropdown over series, statistical annotations and quantile markers.

    Each series is a view of the dropdown. Values outside ``hist_range`` are not drawn but
    **stay in the density denominator**, so a limit never erases tail probability. Quantile
    markers use the full sample and change with the dropdown. Each series is treated as a
    bag of values: the index is ignored.

    Args:
        data: Series, wide DataFrame or ``{label: Series}``.
        title: Figure title.
        subtitle: Muted line under the title.
        series: Subset and order of series.
        bins: Number of bins.
        pooled: Add a first view that stacks every observation with equal weight.
        statistics: Show the statistics annotation (top right).
        display_stats: Which statistics to write; names as in
            :func:`plotters.stats.series_statistics`.
        quantiles: Levels to mark with vertical lines; ``None`` for none.
        normal_reference: Overlay the standard normal N(0, 1), *not* a fitted normal.
        hist_range: ``(low, high)`` drawing window; default is each series' own extent.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.
    """
    frames = pick(as_series_dict(data, _caller="plot_series_distribution"), series)
    values = {str(n): _finite(s).to_numpy() for n, s in frames.items()}
    if pooled and len(values) > 1:
        values = {_POOLED: np.concatenate(list(values.values())), **values}
    stats_table = series_statistics(
        {n: pd.Series(v) for n, v in values.items()}, quantiles=quantiles or ()
    )
    st = _style.get_style()

    views: dict[str, go.Figure] = {}
    for name, v in values.items():
        if v.size == 0:
            raise ValueError(f"series {name!r} has no finite values")
        lo, hi = hist_range if hist_range else (float(v.min()), float(v.max()))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, bins + 1)
        counts, _ = np.histogram(v, edges)
        bin_width = edges[1] - edges[0]
        density = counts / (
            v.size * bin_width
        )  # denominator is the full sample, not just the window
        fig = go.Figure(
            go.Bar(
                x=(edges[:-1] + edges[1:]) / 2, y=density, width=bin_width, name=name, showlegend=False,
                marker={"color": _color.with_alpha(st.accent, 0.72), "line": {"width": 0}},
                hovertemplate="%{x:.4g}: %{y:.4g}<extra></extra>",
            )
        )  # fmt: skip
        if normal_reference:
            grid = np.linspace(lo, hi, 200)
            fig.add_scatter(
                x=grid, y=np.exp(-grid**2 / 2) / np.sqrt(2 * np.pi), mode="lines",
                name="N(0, 1)", line={"color": st.negative, "width": 1.6, "dash": "dash"},
            )  # fmt: skip
        for q in quantiles or ():
            x = float(np.quantile(v, q))
            fig.add_shape(type="line", xref="x", x0=x, x1=x, yref="paper", y0=0, y1=1,
                          line={"color": st.muted, "width": 1, "dash": "dot"})  # fmt: skip
            fig.add_annotation(xref="x", x=x, yref="paper", y=0.0, yanchor="bottom", showarrow=False,
                               text=f"{q:.0%}", font={"size": 9, "color": st.muted})  # fmt: skip
        if statistics:
            row = stats_table.loc[name]
            lines = [f"{k.replace('_', ' ')} {row[k]:.4g}" for k in display_stats if k in row]
            fig.add_annotation(
                xref="paper", x=1, xanchor="right", yref="paper", y=1, yanchor="bottom", align="right",
                showarrow=False, text="<br>".join(lines), font={"size": 10, "color": st.ink_2, "family": st.mono},
            )  # fmt: skip
        fig.update_xaxes(range=[lo, hi])
        fig.update_yaxes(title_text="density")
        views[name] = fig
    return plot_views(views, title, subtitle, height=height, legend=normal_reference, width=width)


def _prepare(
    data: SeriesLike,
    series: Sequence[str] | None,
    normalize: Literal["std", "rms"] | None,
    center: bool,
    cut: int,
) -> dict[str, pd.Series]:
    frames = pick(as_series_dict(data, _caller="analyze_series"), series)
    if normalize is None and not center and not cut:
        return {str(n): s.astype(float) for n, s in frames.items()}
    return normalize_series(frames, normalize, center=center, cut=cut).series


def _paths(frames: Mapping[str, pd.Series], pooled: bool) -> dict[str, dict[str, pd.Series]]:
    """The four time paths per series, plus a pooled one built the documented way.

    The pooled path averages the series *available at each index value* and then accumulates;
    absolute paths average magnitudes before accumulating.
    """
    wide = pd.concat(frames, axis=1)
    out: dict[str, dict[str, pd.Series]] = {
        "values": {},
        "absolute": {},
        "cumulative": {},
        "cumulative_absolute": {},
    }
    columns: dict[str, pd.Series] = {}
    if pooled and len(frames) > 1:
        columns[_POOLED] = wide.mean(axis=1)
    columns.update(frames)
    for name, s in columns.items():
        a = s.abs() if name != _POOLED else wide.abs().mean(axis=1)
        out["values"][name] = s
        out["absolute"][name] = a
        out["cumulative"][name] = s.cumsum()
        out["cumulative_absolute"][name] = a.cumsum()
    return out


def _baselines(reference: SeriesLike | None, names: Sequence[str]) -> dict[str, pd.Series]:
    """``{view name: cumulative baseline}``; a single series is shared by every view."""
    if reference is None:
        return {}
    given = {
        str(k): v.astype(float)
        for k, v in as_series_dict(reference, _caller="analyze_series").items()
    }
    if len(given) == 1 and len(names) > 1 and next(iter(given)) not in names:
        only = next(iter(given.values()))
        return {n: only for n in names if n != _POOLED}
    return {n: given[n] for n in names if n in given}


def _add_baselines(
    fig: go.Figure, baselines: Mapping[str, pd.Series], label: str, names: Sequence[str]
) -> None:
    """Dashed baseline traces, hidden until clicked in the legend, shown only in their view."""
    st = _style.get_style()
    menus = fig.to_plotly_json()["layout"].get("updatemenus")
    menu = menus[0] if menus else None
    for view, s in baselines.items():
        fig.add_scatter(
            x=s.index, y=s.to_numpy(), mode="lines", name=label,
            line={"color": st.muted, "width": 1.6, "dash": "dash"},
            visible="legendonly" if menu is None or view == names[0] else False,
        )  # fmt: skip
    if menu is None:
        return
    for button in menu["buttons"]:
        visible = list(button["args"][0]["visible"])
        visible += ["legendonly" if button["label"] == view else False for view in baselines]
        button["args"][0]["visible"] = visible
    fig.update_layout(updatemenus=[menu])


def _prefix_views(fig: go.Figure, prefix: str) -> None:
    menus = fig.to_plotly_json()["layout"].get("updatemenus")
    if not menus:
        return
    menu = menus[0]
    for button in menu["buttons"]:
        button["label"] = f"{prefix}: {button['label']}"
    fig.update_layout(updatemenus=[menu])


def analyze_series(
    data: SeriesLike,
    title: str = "",
    *,
    series: Sequence[str] | None = None,
    bins: int = 50,
    hist_range: tuple[float, float] | None = None,
    pooled: bool = True,
    normalize: Literal["std", "rms"] | None = None,
    center: bool = False,
    cut: int = 0,
    quantiles: Sequence[float] = (0.01, 0.05, 0.5, 0.95, 0.99),
    display_stats: Sequence[str] = _DEFAULT_DISPLAY,
    thresholds: Sequence[float] | None = (1, 2, 3, 4),
    metrics: Mapping[str, Callable[[pd.Series], float]] | None = None,
    reference_series: SeriesLike | None = None,
    reference_label: str = "reference",
    normal_reference: bool = False,
    turnover_ratio: Callable[[pd.Series], float] | None = None,
    value_label: str = "",
    view_prefix: str = "",
    tail_normalize: Literal["std", "rms"] | None = "rms",
    tail_center: bool = False,
    tail_ratios: Sequence[tuple[float, float]] = ((3, 1),),
    energy_fractions: Sequence[float] = (0.01,),
    acf_lags: Sequence[int] = (1, 5, 21),
    rolling_window: int | None = 21,
    extreme_multiples: Sequence[float] = (3, 4),
) -> dict[str, Any]:
    """Statistics and five figures for a set of series, in one call.

    The figures are the values, absolute values, cumulative sums, cumulative absolute sums
    and a histogram. Each has a dropdown with one view per series, plus a pooled view.

    A pooled **distribution** and a pooled **path** are different quantities: the histogram
    stacks observations with equal weight per observation, while the time plots average the
    series available at each index value. Absolute paths average magnitudes *before*
    accumulating. Set ``pooled=False`` when combining the inputs' units makes no sense.

    Args:
        data: Series, wide DataFrame or ``{label: Series}``.
        title: Prefix for the figure titles.
        series: Subset and order of series.
        bins: Histogram bins.
        hist_range: ``(low, high)`` window drawn by the histogram, in the units after
            ``normalize``; ``None`` shows each series' own extent. Values outside the window
            stay in the density denominator and in ``stats``.
        pooled: Add a pooled view and a pooled statistics row.
        normalize: ``"std"`` or ``"rms"`` scales each series by its own full retained sample;
            ``None`` keeps the units. Descriptive normalisation, not a causal transform.
        center: Also subtract each series' mean (needs ``normalize`` to be meaningful).
        cut: Leading rows to drop after sorting each input's index.
        quantiles: Quantile levels for the table and the histogram markers.
        display_stats: Statistics written on the histogram annotation.
        thresholds: Tail cut-offs in units of ``tail_normalize`` (own RMS by default), see
            :func:`plotters.stats.series_statistics`.
        metrics: ``{name: callable}`` extra scalar statistics.
        reference_series: Series, wide frame or ``{label: Series}`` of baselines **in the same units
            as the values** (no rescaling is applied), matched to the inputs by name; one series is
            used for every input. Each baseline is a cumulative path, drawn dashed and **hidden
            until clicked in the legend**, on the ``cumulative`` figure.
        reference_label: Legend name of that baseline.
        normal_reference: Overlay N(0, 1) on the histogram: a descriptive reference, not a fit or a
            normality test.
        turnover_ratio: ``callable(series) -> float`` replacing :func:`series_turnover_ratio` in the
            statistics and in the dropdown text.
        value_label: Quantity and units for the axes (y of the time figures, x of the histogram).
        view_prefix: Text before each dropdown entry, e.g. ``"Ticker"`` gives ``Ticker: AAPL``.
        tail_normalize: ``"rms"`` / ``"std"`` scales each series by its own full retained sample
            before the tail columns; see :func:`plotters.stats.series_statistics`, as for the next
            four.
        tail_center: Subtract the mean before tail scaling.
        tail_ratios: ``((a, b), ...)`` adds ``tail_ratio_<a>_<b>`` columns.
        energy_fractions: Adds ``squared_tail_share_<pct>pct`` columns.
        acf_lags: Adds ``acf_``, ``abs_acf_`` and ``squared_acf_`` columns at these lags.
        rolling_window: Adds ``rolling_rms_cv_<w>``.
        extreme_multiples: Adds ``extreme_frequency_<k>rms`` (share beyond ``k`` times the own RMS).

    The defaults give the table: ``n, mean, std, rms, mean_absolute, mean_over_std, skewness,
    excess_kurtosis, median, min, max``, ``q01 q05 q50 q95 q99``,
    ``tail_probability_{1,2,3,4}rms`` and ``tail_count_{1,2,3,4}rms`` (P(|z| > t) for z = x / own
    RMS), ``tail_ratio_3rms_1rms``, ``squared_tail_share_1pct``, ``extreme_frequency_{3,4}rms``,
    ``acf_`` / ``abs_acf_`` / ``squared_acf_`` at lags 1, 5, 21, ``rolling_rms_cv_21`` and
    ``turnover_ratio``. Pass ``()`` / ``None`` to drop an optional group.

    Returns:
        ``{"data": {label: Series}, "stats": DataFrame, "figures": {name: Figure}}`` where
        the figures are ``values``, ``absolute``, ``cumulative``, ``cumulative_absolute`` and
        ``histogram``.

    Example:
        >>> out = pl.analyze_series(returns, bins=100, normalize="std")
        >>> out["stats"]
        >>> pl.show_all(out["figures"])
    """
    frames = _prepare(data, series, normalize, center, cut)
    tail = {
        "tail_normalize": tail_normalize,
        "tail_center": tail_center,
        "tail_ratios": tail_ratios,
        "energy_fractions": energy_fractions,
        "extreme_multiples": extreme_multiples,
    }
    stats = series_statistics(
        frames, quantiles=quantiles, thresholds=thresholds, metrics=metrics, turnover=turnover_ratio,
        acf_lags=acf_lags, rolling_window=rolling_window, **tail,
    )  # fmt: skip
    if pooled and len(frames) > 1:
        row = _pooled_statistics(
            frames, quantiles, thresholds, metrics, stats, tail, turnover_ratio
        )
        stats = pd.concat([row.to_frame(_POOLED).T, stats])

    paths = _paths(frames, pooled)
    names = list(next(iter(paths.values())))
    views = {n: [n] for n in names}
    prefix = f"{title}: " if title else ""
    labels = {
        "values": "values",
        "absolute": "absolute values",
        "cumulative": "cumulative sum",
        "cumulative_absolute": "cumulative absolute sum",
    }
    axis_titles = {
        "values": value_label,
        "absolute": f"|{value_label}|" if value_label else "",
        "cumulative": f"cumulative {value_label}" if value_label else "",
        "cumulative_absolute": f"cumulative |{value_label}|" if value_label else "",
    }
    baselines = _baselines(reference_series, names)
    turnover_fn = turnover_ratio or series_turnover_ratio
    figures: dict[str, go.Figure] = {}
    for key, text in labels.items():
        with_baseline = key == "cumulative" and bool(baselines)
        fig = plot_series(
            paths[key], prefix + text, views=views if len(names) > 1 else None,
            ylabel=axis_titles[key], legend=with_baseline,
        )  # fmt: skip
        if with_baseline:
            _add_baselines(fig, baselines, reference_label, names)
        if len(names) > 1 and key == "values":
            annotate_views(
                fig, {n: f"turnover {turnover_fn(paths['values'][n]):.4g}" for n in names}
            )
        figures[key] = fig
    figures["histogram"] = plot_series_distribution(
        frames, prefix + "distribution", bins=bins, hist_range=hist_range, pooled=pooled,
        quantiles=quantiles, display_stats=display_stats, normal_reference=normal_reference,
    )  # fmt: skip
    if value_label:
        figures["histogram"].update_xaxes(title_text=value_label)
    if view_prefix:
        for fig in figures.values():
            _prefix_views(fig, view_prefix)
    first = [c for c in ("n", *display_stats) if c in stats.columns]
    stats = stats[[*first, *[c for c in stats.columns if c not in first]]]
    return {"data": frames, "stats": stats, "figures": figures}


def compare_series(
    data: SeriesLike,
    title: str = "",
    *,
    reference: str | None = None,
    series: Sequence[str] | None = None,
    align: Literal["inner", "outer"] = "inner",
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> dict[str, Any]:
    """Several series on the same dates: cumulative paths and differences against one.

    Differences are **reference minus each other series**. More series do not create all
    pairs: every comparison is against the one reference.

    Args:
        data: Series, wide DataFrame or ``{label: Series}``.
        title: Prefix for the figure titles.
        reference: Name of the reference series; the first series when ``None``.
        series: Subset and order of series (the reference must be in it).
        align: ``"inner"`` uses the observations that are common and finite for all inputs.
            ``"outer"`` keeps gaps, and each difference uses its own available pairs.
        quantiles: Levels for the statistics table of the differences.

    Returns:
        ``{"data", "cumulative", "difference", "stats", "figures"}``: aligned inputs, their
        cumulative sums, per-period differences, statistics of the differences, and the
        ``cumulative`` and ``difference`` (cumulative difference) figures.

    Raises:
        KeyError: ``reference`` is not one of the series.
        ValueError: fewer than two series, or ``align`` is not ``"inner"`` / ``"outer"``.
    """
    if align not in ("inner", "outer"):
        raise ValueError(f"align must be 'inner' or 'outer', got {align!r}")
    frames = pick(as_series_dict(data, _caller="compare_series"), series)
    if len(frames) < 2:
        raise ValueError("compare_series needs at least two series")
    ref = reference if reference is not None else next(iter(frames))
    if ref not in frames:
        raise KeyError(f"reference {ref!r} not in {list(frames)}")

    wide = pd.concat({str(k): v.astype(float) for k, v in frames.items()}, axis=1).sort_index()
    wide = wide.where(np.isfinite(wide))
    if align == "inner":
        wide = wide.dropna()
    others = [c for c in wide.columns if c != ref]
    difference = pd.DataFrame({c: wide[ref] - wide[c] for c in others})
    cumulative = wide.cumsum()
    prefix = f"{title}: " if title else ""
    figures = {
        "cumulative": plot_series(cumulative, prefix + "cumulative paths", ylabel="cumulative sum"),
        "difference": plot_series(
            difference.cumsum(), prefix + f"cumulative difference ({ref} minus others)",
            ylabel="cumulative difference",
        ),
    }  # fmt: skip
    return {
        "data": wide,
        "cumulative": cumulative,
        "difference": difference,
        "stats": series_statistics(difference, quantiles=quantiles),
        "figures": figures,
    }
