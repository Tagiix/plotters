"""Time-series charts. All convention A (Series / wide DataFrame / dict) except
:func:`plot_bars_with_line` (a Series of bars and a convention-A line) and
:func:`plot_calendar_heatmap` (one Series with a DatetimeIndex).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import _color
from . import style as _style
from ._errors import convention_error
from ._types import MAX_SERIES, MAX_VISIBLE, SeriesLike
from .frames import as_series_dict, pick
from .layout import HEIGHT, apply_log, finish, prefix_views

__all__ = [
    "plot_series",
    "plot_bars_with_line",
    "plot_cumulative",
    "plot_drawdown",
    "plot_stacked",
    "plot_rolling",
    "plot_line_band",
    "plot_calendar_heatmap",
]

Visibility = bool | str  # True, False or "legendonly"
_PERCENT_TICKS = ".2~%"


def _x(index: pd.Index) -> pd.Index:
    """Index plotly can plot: a PeriodIndex becomes timestamps."""
    return index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index


def _initial_visibility(count: int, drawn: int) -> list[Visibility]:
    """``True`` for the first ``drawn`` series, ``"legendonly"`` for the rest (one click to add).

    ``drawn=0`` means the default (``MAX_VISIBLE``); it never hides everything.
    """
    if drawn < 0:
        raise ValueError(f"drawn must be 0 (the default) or positive, got {drawn}")
    limit = drawn or MAX_VISIBLE
    return [True if i < limit else "legendonly" for i in range(count)]


def _line(i: int, *, vary_dash: bool = False, width: float | None = None) -> dict[str, object]:
    dash, w = _style.line_style(i, vary_dash=vary_dash)
    return {"color": _style.color(i), "dash": dash, "width": width or w}


def _isolated(y: Any) -> np.ndarray:
    """Finite values with no finite neighbour: a line draws nothing for them."""
    ok = np.isfinite(np.asarray(y, dtype=float))
    left = np.concatenate(([False], ok[:-1]))
    right = np.concatenate((ok[1:], [False]))
    return ok & ~left & ~right


def _resolve_visibility(
    names: Sequence[str],
    views: Mapping[str, Sequence[str]] | None,
    drawn: int,
) -> dict[str, list[Visibility]]:
    """Decide ``visible`` for every series in every view.

    Returns ``{view_name: [visible per series]}``; with no views a single ``""`` view.
    A series is ``True`` (drawn), ``"legendonly"`` (greyed out, one click to add) or
    ``False`` (not in this view, no legend entry).

    * Without views, the first ``drawn`` series (input order) are drawn.
    * With views, a series named in no view is a *common baseline*: drawn in all views and
      counted first towards ``drawn``, so a pooled line sits behind each per-ticker
      view without being repeated in every list. Past ``MAX_SERIES`` such series the rule
      is dropped and the views are an explicit allowlist, otherwise a caller showing its
      best 8 of 42 would leave 34 "unnamed" series drawn everywhere.
    """
    if not views:
        return {"": _initial_visibility(len(names), drawn)}

    unknown = sorted({n for members in views.values() for n in members if n not in names})
    if unknown:
        raise KeyError(f"views name series that are not in the data: {unknown}")

    out: dict[str, list[Visibility]] = {}
    for view, ordered in _view_orders(names, views).items():
        shown = {n: (True if i < drawn else "legendonly") for i, n in enumerate(ordered)}
        out[view] = [shown.get(n, False) for n in names]
    return out


def _view_orders(names: Sequence[str], views: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """The series of each view in draw order: common baselines first, then its own members."""
    named = {n for members in views.values() for n in members}
    baselines = [n for n in names if n not in named]
    if len(baselines) > MAX_SERIES:
        baselines = []
    return {
        view: baselines + [m for m in members if m not in baselines]
        for view, members in views.items()
    }


def _color_slots(names: Sequence[str], views: Mapping[str, Sequence[str]] | None) -> dict[str, int]:
    """Palette slot of each series. With views, every view reuses the slots from the start, so
    the dropdown does not walk down the palette; a series keeps one slot across the views it is in."""
    if not views:
        return {n: i for i, n in enumerate(names)}
    slots: dict[str, int] = {}
    for ordered in _view_orders(names, views).values():
        taken = {slots[n] for n in ordered if n in slots}
        free = (i for i in range(len(names) + 1) if i not in taken)
        for n in ordered:
            if n not in slots:
                slots[n] = next(free)
                taken.add(slots[n])
    for n in names:  # dropped baselines or series in no view
        slots.setdefault(n, len(slots))
    return slots


def _add_reference_lines(
    fig: go.Figure,
    bands: Mapping[str, float] | None,
    markers: Mapping[str, Any] | None,
    logy: bool,
    logx: bool,
    spans: Mapping[str, tuple[Any, Any]] | None = None,
) -> None:
    st = _style.get_style()
    for label, (start, end) in (spans or {}).items():
        fig.add_shape(
            type="rect", xref="x", x0=start, x1=end, yref="paper", y0=0, y1=1, layer="below",
            fillcolor=_color.with_alpha(st.muted, 0.14), line={"width": 0},
        )  # fmt: skip
        fig.add_annotation(
            xref="x", x=math.log10(start) if logx else start, xanchor="left", yref="paper", y=1,
            yanchor="top", text=label, showarrow=False, xshift=4,
            font={"size": 10, "color": st.muted},
        )  # fmt: skip
    # Shapes take data units on a log axis, but annotations take log10 (checked on plotly 7.1).
    for label, y in (bands or {}).items():
        fig.add_shape(
            type="line",
            xref="paper",
            x0=0,
            x1=1,
            yref="y",
            y0=y,
            y1=y,
            line={"color": st.muted, "width": 1, "dash": "dot"},
        )
        fig.add_annotation(
            xref="paper",
            x=1,
            xanchor="right",
            yref="y",
            y=math.log10(y) if logy else y,
            yanchor="bottom",
            text=label,
            showarrow=False,
            font={"size": 10, "color": st.muted},
        )
    for label, x in (markers or {}).items():
        fig.add_shape(
            type="line",
            xref="x",
            x0=x,
            x1=x,
            yref="paper",
            y0=0,
            y1=1,
            line={"color": st.muted, "width": 1, "dash": "dot"},
        )
        fig.add_annotation(
            xref="x",
            x=math.log10(x) if logx else x,
            xanchor="left",
            yref="paper",
            y=1,
            yanchor="top",
            text=label,
            showarrow=False,
            font={"size": 10, "color": st.muted},
        )


def _positive_bounds(frames: Mapping[str, pd.Series]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(s, dtype=float) for s in frames.values()])
    values = values[np.isfinite(values)]
    if values.size == 0 or (values <= 0).any():
        raise ValueError("a log axis needs strictly positive data")
    return float(values.min()) / 1.15, float(values.max()) * 1.15


def plot_series(
    data: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    series: Sequence[str] | None = None,
    percent: bool = False,
    logy: bool = False,
    logx: bool = False,
    slider: bool = False,
    range_buttons: bool = False,
    fill: bool = False,
    bands: Mapping[str, float] | None = None,
    markers: Mapping[str, Any] | None = None,
    spans: Mapping[str, tuple[Any, Any]] | None = None,
    step: bool = False,
    views: Mapping[str, Sequence[str]] | None = None,
    view_prefix: str = "",
    drawn: int = MAX_VISIBLE,
    xlabel: str = "",
    ylabel: str = "",
    height: int = HEIGHT,
    width: int | None = None,
    legend: bool | None = None,
) -> go.Figure:
    """One line per series: the workhorse time-series chart.

    There is no limit on the number of series. Every series is kept, but only the first
    ``drawn`` start drawn; the rest are greyed out in the legend, one click to add.

    Args:
        data: Series, wide DataFrame (one series per column) or ``{label: Series}``.
        title: Figure title.
        subtitle: Muted line under the title.
        series: Subset *and order* of series to draw; the order assigns colour slots.
        percent: Format the y axis as a percentage (values are fractions: 0.01 is 1%).
        logy: Logarithmic y axis with readable ticks (data must be > 0).
        logx: Logarithmic x axis (numeric index > 0).
        slider: Add a range slider under the x axis.
        range_buttons: Add quick date-range buttons (1m, 3m, 6m, YTD, 1y, all) at the top right
            of the plot. Needs a datetime x axis.
        fill: Fill each line down to zero.
        bands: ``{label: y}`` horizontal reference lines. Scalars, not series.
        markers: ``{label: x}`` vertical reference lines, e.g. events.
        spans: ``{label: (start, end)}`` shaded vertical bands behind the lines, e.g. a recession
            or an outage window. Drawn below the data so the lines stay readable.
        step: Draw each series as a step function (the value holds until the next observation)
            instead of joining points with straight lines.
        views: ``{view_name: [series names]}``. Adds a dropdown at the top left that swaps
            between named subsets of the series already passed. Every series is drawn once
            and the menu toggles ``visible``, so an exported HTML keeps working with no
            server. A series named in no view stays visible in all of them.
        view_prefix: Text before each dropdown entry, e.g. ``"Ticker"`` gives ``Ticker: AAPL``.
        drawn: How many series start drawn (default 20; ``0`` also means the default, it never
            hides everything). The rest are greyed out in the legend.
        xlabel: X axis title.
        ylabel: Y axis title.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.
        legend: ``None`` shows it for more than one series; ``True`` / ``False`` force it.

    Returns:
        A plotly ``Figure``. Nothing is shown.

    Raises:
        KeyError: ``series`` or ``views`` names a label that is not in ``data``.
        ValueError: a log axis was requested but the data is not strictly positive, or
            ``drawn`` is negative.

    Example:
        >>> pl.plot_series(returns, "Daily returns", percent=True)
    """
    frames = pick(as_series_dict(data, _caller="plot_series"), series)
    names = list(frames)
    visibility = _resolve_visibility([str(n) for n in names], views, drawn or MAX_VISIBLE)
    _initial_visibility(0, drawn)  # validates drawn
    first = next(iter(visibility.values()))
    st = _style.get_style()

    slots = _color_slots([str(n) for n in names], views)
    fig = go.Figure()
    for i, (name, s) in enumerate(frames.items()):
        line = _line(slots[str(name)])
        trace = go.Scatter(
            x=_x(s.index),
            y=s.to_numpy(),
            name=str(name),
            mode="lines",
            line=line,
            line_shape="hv" if step else "linear",
            visible=first[i],
        )
        if fill:
            trace.update(fill="tozeroy", fillcolor=_color.with_alpha(line["color"], 0.12))  # type: ignore[arg-type]
        isolated = _isolated(s.to_numpy())
        if isolated.any():  # a lone point has no line to draw it
            trace.update(
                mode="lines+markers",
                marker={"size": np.where(isolated, 7, 0).tolist(), "color": line["color"]},
            )
        fig.add_trace(trace)

    _add_reference_lines(fig, bands, markers, logy, logx, spans)

    if views:
        fig.update_layout(
            updatemenus=[
                {
                    "type": "dropdown",
                    "direction": "down",
                    "x": 0,
                    "xanchor": "left",
                    "y": 1.0,
                    "yanchor": "bottom",
                    "pad": {"t": 0, "b": 6},
                    "bgcolor": st.surface,
                    "bordercolor": st.axis,
                    "font": {"color": st.ink_2, "size": 11},
                    "buttons": [
                        {"label": view, "method": "update", "args": [{"visible": vis}]}
                        for view, vis in visibility.items()
                    ],
                }
            ]
        )
        prefix_views(fig, view_prefix)

    if percent:
        fig.update_yaxes(tickformat=_PERCENT_TICKS)
        fig.update_traces(hovertemplate="%{y:.2%}", selector={"type": "scatter"})
    if range_buttons:
        fig.update_xaxes(
            rangeselector={
                "buttons": [
                    {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
                    {"count": 3, "label": "3m", "step": "month", "stepmode": "backward"},
                    {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                    {"label": "YTD", "step": "year", "stepmode": "todate"},
                    {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                    {"label": "all", "step": "all"},
                ],
                "x": 1,
                "xanchor": "right",
                "y": 1.0,
                "yanchor": "bottom",
                "bgcolor": st.surface,
                "activecolor": st.grid,
                "bordercolor": st.axis,
                "font": {"size": 11, "color": st.ink_2},
            }
        )
    if slider:
        fig.update_xaxes(rangeslider={"visible": True, "thickness": 0.06})
    if logy:
        apply_log(fig, "y", *_positive_bounds(frames))
    if logx:
        low = min(float(np.min(np.asarray(_x(s.index), dtype=float))) for s in frames.values())
        high = max(float(np.max(np.asarray(_x(s.index), dtype=float))) for s in frames.values())
        apply_log(fig, "x", low / 1.1, high * 1.1)

    return finish(
        fig,
        title,
        subtitle,
        xlabel=xlabel,
        ylabel=ylabel,
        height=height,
        width=width,
        slider=slider,
        legend=legend,
        menu=bool(views) or range_buttons,
    )


def plot_bars_with_line(
    bars: pd.Series,
    line: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    xlabel: str = "",
    ylabel: str = "",
    percent: bool = False,
    height: int = HEIGHT,
    width: int | None = None,
) -> go.Figure:
    """Realised values as bars, predictions as lines, on one axis: did it track or lag?

    Convention C: two positional arguments.

    Args:
        bars: A single Series of realised values. A dict or DataFrame raises.
        line: The prediction(s): Series, wide DataFrame or ``{label: Series}``. A dict
            compares several predictions against the same bars.
        title: Figure title.
        subtitle: Muted line under the title.
        xlabel: X axis title.
        ylabel: Y axis title.
        percent: Format the y axis as a percentage.
        height: Figure height in px.
        width: Figure width in px.

    Raises:
        ConventionError: ``bars`` is not a Series.
    """
    if not isinstance(bars, pd.Series):
        raise convention_error("plot_bars_with_line", "C", bars, "bars must be a single Series.")
    lines = as_series_dict(line, _caller="plot_bars_with_line")
    st = _style.get_style()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=_x(bars.index),
            y=bars.to_numpy(),
            name=str(bars.name or "realised"),
            marker={"color": _color.with_alpha(st.muted, 0.45), "line": {"width": 0}},
        )
    )
    for i, (name, s) in enumerate(lines.items()):
        fig.add_trace(
            go.Scatter(x=_x(s.index), y=s.to_numpy(), name=str(name), mode="lines", line=_line(i))
        )
    if percent:
        fig.update_yaxes(tickformat=_PERCENT_TICKS)
    return finish(
        fig, title, subtitle, xlabel=xlabel, ylabel=ylabel, height=height, width=width, legend=True
    )


def _wealth(s: pd.Series, compound: bool) -> pd.Series:
    """Wealth curve from per-period values, starting at 1."""
    r = s.astype(float).fillna(0.0)
    return (1 + r).cumprod() if compound else 1 + r.cumsum()


def plot_cumulative(
    data: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    series: Sequence[str] | None = None,
    compound: bool = False,
    logy: bool = False,
    drawn: int = MAX_VISIBLE,
    xlabel: str = "",
    ylabel: str = "Growth of 1",
    height: int = HEIGHT,
    width: int | None = None,
    legend: bool | None = None,
) -> go.Figure:
    """Wealth curve from per-period values, starting at 1.

    Args:
        data: Per-period values (fractions, 0.01 is 1%) as Series, wide DataFrame or dict.
        title: Figure title.
        subtitle: Muted line under the title.
        series: Subset and order of series.
        compound: ``True`` for reinvested growth (``cumprod(1 + r)``); the default adds the
            values (``1 + cumsum(r)``). Missing values count as zero.
        logy: Log y axis; use with ``compound=True``.
        drawn: How many series start drawn (default 20; ``0`` also means the default). The rest
            are greyed out in the legend, one click to add.
        xlabel: X axis title.
        ylabel: Y axis title.
        height: Figure height in px.
        width: Figure width in px.
        legend: ``None`` shows it for more than one series.

    Raises:
        ValueError: ``drawn`` is negative.
    """
    frames = pick(as_series_dict(data, _caller="plot_cumulative"), series)
    flags = _initial_visibility(len(frames), drawn)
    curves = {n: _wealth(s, compound) for n, s in frames.items()}
    fig = go.Figure()
    for i, (name, s) in enumerate(curves.items()):
        fig.add_trace(
            go.Scatter(
                x=_x(s.index),
                y=s.to_numpy(),
                name=str(name),
                mode="lines",
                line=_line(i),
                visible=flags[i],
            )
        )
    if logy:
        apply_log(fig, "y", *_positive_bounds(curves))
    return finish(
        fig,
        title,
        subtitle,
        xlabel=xlabel,
        ylabel=ylabel,
        height=height,
        width=width,
        legend=legend,
    )


def plot_drawdown(
    data: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    series: Sequence[str] | None = None,
    compound: bool = True,
    drawn: int = MAX_VISIBLE,
    xlabel: str = "",
    ylabel: str = "Drawdown",
    height: int = HEIGHT,
    width: int | None = None,
    legend: bool | None = None,
) -> go.Figure:
    """Drop from the running peak, filled because the area *is* the quantity.

    Args:
        data: Per-period values (fractions) as Series, wide DataFrame or dict.
        title: Figure title.
        subtitle: Muted line under the title.
        series: Subset and order of series.
        compound: Peak-relative drop of the compounded curve (percent). ``False`` measures
            the drop of the summed curve in the data's own units.
        drawn: How many series start drawn (default 20; ``0`` also means the default). The rest
            are greyed out in the legend, one click to add.
        xlabel: X axis title.
        ylabel: Y axis title.
        height: Figure height in px.
        width: Figure width in px.
        legend: ``None`` shows it for more than one series.

    Raises:
        ValueError: ``drawn`` is negative.
    """
    frames = pick(as_series_dict(data, _caller="plot_drawdown"), series)
    flags = _initial_visibility(len(frames), drawn)
    fig = go.Figure()
    for i, (name, s) in enumerate(frames.items()):
        wealth = _wealth(s, compound)
        dd = wealth / wealth.cummax() - 1 if compound else wealth - wealth.cummax()
        line = _line(i)
        fig.add_trace(
            go.Scatter(
                x=_x(dd.index),
                y=dd.to_numpy(),
                name=str(name),
                mode="lines",
                line=line,
                fill="tozeroy",
                fillcolor=_color.with_alpha(line["color"], 0.18),  # type: ignore[arg-type]
                visible=flags[i],
            )
        )
    if compound:
        fig.update_yaxes(tickformat=_PERCENT_TICKS)
    return finish(
        fig,
        title,
        subtitle,
        xlabel=xlabel,
        ylabel=ylabel,
        height=height,
        width=width,
        legend=legend,
    )


def plot_stacked(
    data: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    series: Sequence[str] | None = None,
    normalize: bool = False,
    xlabel: str = "",
    ylabel: str = "",
    height: int = HEIGHT,
    width: int | None = None,
) -> go.Figure:
    """Parts of a whole, stacked. Refuses negative parts: a stack of them means nothing.

    Args:
        data: Non-negative parts as Series, wide DataFrame or dict.
        title: Figure title.
        subtitle: Muted line under the title.
        series: Subset and order of series.
        normalize: Show each part as a share of the total at that index (sums to 100%).
        xlabel: X axis title.
        ylabel: Y axis title.
        height: Figure height in px.
        width: Figure width in px.

    Raises:
        ValueError: a part is negative.
    """
    frames = pick(as_series_dict(data, _caller="plot_stacked"), series)
    for name, s in frames.items():
        if (s.dropna() < 0).any():
            raise ValueError(f"plot_stacked needs non-negative parts; {name!r} has negative values")
    fig = go.Figure()
    for i, (name, s) in enumerate(frames.items()):
        color = _style.color(i)
        fig.add_trace(
            go.Scatter(
                x=_x(s.index),
                y=s.to_numpy(),
                name=str(name),
                mode="lines",
                line={"color": color, "width": 0.8},
                stackgroup="one",
                groupnorm="percent" if normalize else "",
                fillcolor=_color.with_alpha(color, 0.75),
            )
        )
    if normalize:
        fig.update_yaxes(ticksuffix="%")
    return finish(
        fig, title, subtitle, xlabel=xlabel, ylabel=ylabel, height=height, width=width, legend=True
    )


_ROLLING = ("mean", "std", "var", "median", "min", "max", "sum", "skew", "kurt")


def plot_rolling(
    data: SeriesLike,
    window: int,
    title: str = "",
    subtitle: str = "",
    *,
    how: Literal["mean", "std", "var", "median", "min", "max", "sum", "skew", "kurt"] = "mean",
    series: Sequence[str] | None = None,
    drawn: int = MAX_VISIBLE,
    xlabel: str = "",
    ylabel: str = "",
    height: int = HEIGHT,
    width: int | None = None,
    legend: bool | None = None,
) -> go.Figure:
    """A rolling statistic drawn over the raw series in lighter ink.

    One y axis, always: the raw series and the statistic share it, so this reads best for
    statistics in the data's own units (mean, median, min, max). For ``"std"``, ``"var"``
    or ``"skew"`` the two live on different scales; chart those with :func:`plot_series`
    on the computed rolling series instead.

    Args:
        data: Series, wide DataFrame or dict.
        window: Window length in observations.
        title: Figure title.
        subtitle: Muted line under the title.
        how: Statistic: ``"mean"``, ``"std"``, ``"var"``, ``"median"``, ``"min"``,
            ``"max"``, ``"sum"``, ``"skew"`` or ``"kurt"``.
        series: Subset and order of series.
        drawn: How many series start drawn (default 20; ``0`` also means the default). The rest
            are greyed out in the legend, one click to add.
        xlabel: X axis title.
        ylabel: Y axis title.
        height: Figure height in px.
        width: Figure width in px.
        legend: ``None`` shows it for more than one series.

    Raises:
        ValueError: unknown ``how``, a window below 2, or ``drawn`` negative.
    """
    if how not in _ROLLING:
        raise ValueError(f"how must be one of {list(_ROLLING)}, got {how!r}")
    if window < 2:
        raise ValueError("window must be at least 2")
    frames = pick(as_series_dict(data, _caller="plot_rolling"), series)
    flags = _initial_visibility(len(frames), drawn)
    fig = go.Figure()
    for i, (name, s) in enumerate(frames.items()):
        color = _style.color(i)
        dash = _line(i)["dash"]
        rolled = getattr(s.astype(float).rolling(window), how)()
        fig.add_trace(
            go.Scatter(
                x=_x(s.index),
                y=s.to_numpy(),
                name=str(name),
                legendgroup=str(name),
                showlegend=False,
                mode="lines",
                line={"color": _color.with_alpha(color, 0.35), "width": 1},
                hoverinfo="skip",
                visible=flags[i],
            )
        )
        fig.add_trace(
            go.Scatter(
                x=_x(rolled.index),
                y=rolled.to_numpy(),
                name=str(name),
                legendgroup=str(name),
                mode="lines",
                line={"color": color, "width": 2.2, "dash": dash},
                visible=flags[i],
            )
        )
    sub = subtitle or f"rolling {how}, window {window}; raw series in lighter ink"
    return finish(
        fig, title, sub, xlabel=xlabel, ylabel=ylabel, height=height, width=width, legend=legend
    )


def plot_line_band(
    data: SeriesLike,
    lower: SeriesLike,
    upper: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    series: Sequence[str] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    height: int = HEIGHT,
    width: int | None = None,
) -> go.Figure:
    """A line with an uncertainty ribbon behind it.

    ``lower`` and ``upper`` take the same shape as ``data`` and must hold the same labels.

    Args:
        data: The central line(s): Series, wide DataFrame or ``{label: Series}``.
        lower: Lower edge of each ribbon.
        upper: Upper edge of each ribbon.
        title: Figure title.
        subtitle: Muted line under the title.
        series: Subset and order of series.
        xlabel: X axis title.
        ylabel: Y axis title.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        KeyError: ``lower`` / ``upper`` lack a label present in ``data``.
    """
    centre = pick(as_series_dict(data, _caller="plot_line_band"), series)
    lo = as_series_dict(lower, _caller="plot_line_band")
    hi = as_series_dict(upper, _caller="plot_line_band")
    for name in centre:
        for which, d in (("lower", lo), ("upper", hi)):
            if name not in d:
                raise KeyError(f"{which} has no series {name!r}; it has {list(d)}")
    fig = go.Figure()
    for i, (name, s) in enumerate(centre.items()):
        color = _style.color(i)
        fig.add_scatter(x=hi[name].index, y=hi[name].to_numpy(), mode="lines", line={"width": 0}, hoverinfo="skip",
                        showlegend=False, legendgroup=str(name))  # fmt: skip
        fig.add_scatter(x=lo[name].index, y=lo[name].to_numpy(), mode="lines", line={"width": 0}, fill="tonexty",
                        fillcolor=_color.with_alpha(color, 0.18), hoverinfo="skip", showlegend=False,
                        legendgroup=str(name))  # fmt: skip
        fig.add_scatter(x=s.index, y=s.to_numpy(), mode="lines", name=str(name), legendgroup=str(name),
                        line={"color": color, "width": 2})  # fmt: skip
    return finish(
        fig,
        title,
        subtitle,
        xlabel=xlabel,
        ylabel=ylabel,
        height=height,
        legend=len(centre) > 1,
        width=width,
    )


def plot_calendar_heatmap(
    data: pd.Series,
    title: str = "",
    subtitle: str = "",
    *,
    fmt: str = ".4g",
    height: int = 260,
    width: int | None = None,
) -> go.Figure:
    """A daily series as a weekday-by-week grid, shaded by value (like a contribution calendar).

    Args:
        data: Series with a DatetimeIndex; several values on one day are summed.
        title: Figure title.
        subtitle: Muted line under the title.
        fmt: Format spec for the hover value.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ValueError: the index is not a DatetimeIndex.
    """
    if not isinstance(data, pd.Series) or not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("plot_calendar_heatmap needs a Series with a DatetimeIndex")
    daily = data.astype(float).groupby(data.index.normalize()).sum()
    full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    days = pd.DatetimeIndex(full.index)
    week_start = days - pd.to_timedelta(days.day_of_week, unit="D")
    grid = pd.DataFrame({"week": week_start, "day": days.day_of_week, "v": full.to_numpy()})
    z = grid.pivot(index="day", columns="week", values="v").reindex(range(7))
    st = _style.get_style()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dates = [
        [(pd.Timestamp(w) + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for w in z.columns]
        for d in z.index
    ]
    fig = go.Figure(go.Heatmap(
        z=z.to_numpy(), x=z.columns, y=names, colorscale=_style.sequential_scale(), xgap=2, ygap=2, customdata=dates,
        hovertemplate="%{customdata}: %{z:" + fmt + "}<extra></extra>",
        colorbar={"thickness": 10, "outlinewidth": 0, "tickfont": {"size": 10, "color": st.muted}},
    ))  # fmt: skip
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(showgrid=False, tickformat="%b", dtick="M1")
    return finish(fig, title, subtitle, height=height, legend=False, width=width)
