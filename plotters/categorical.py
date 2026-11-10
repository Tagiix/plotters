"""Basic categorical and relationship charts.

Convention E takes ``{label: number}``, or a Series indexed by label, so the output of
``value_counts()`` or a plain count dictionary plots directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import _color
from . import style as _style
from ._errors import convention_error
from ._types import MAX_SERIES, SeriesLike
from .frames import _one_column, _require_columns, as_series_dict, as_value_series, pick
from .layout import HEIGHT, finish

__all__ = [
    "plot_bars",
    "plot_pareto",
    "plot_counts",
    "plot_grouped_bars",
    "plot_lollipop",
    "plot_dumbbell",
    "plot_waterfall",
    "plot_line_band",
    "plot_regression",
    "plot_scatter_matrix",
    "plot_calendar_heatmap",
]

Sort = bool | Literal["ascending", "descending"]


def _ordered(s: pd.Series, sort: Sort, top: int | None) -> pd.Series:
    if sort is True or sort == "descending":
        s = s.sort_values(ascending=False, kind="stable")
    elif sort == "ascending":
        s = s.sort_values(ascending=True, kind="stable")
    return s.head(top) if top is not None else s


def _row_height(n: int) -> int:
    return int(min(1100, max(260, 130 + 30 * n)))


def plot_bars(
    data: Mapping[str, float] | pd.Series,
    title: str = "",
    subtitle: str = "",
    *,
    sort: Sort = True,
    top: int | None = None,
    horizontal: bool = True,
    normalize: bool = False,
    error: Mapping[str, float]
    | pd.Series
    | tuple[Mapping[str, float], Mapping[str, float]]
    | None = None,
    reference: float | None = None,
    fmt: str = ",.4g",
    xlabel: str = "",
    ylabel: str = "",
    height: int | None = None,
) -> go.Figure:
    """Bars from a ``{label: number}`` dict or a Series, with the value written at each tip.

    One hue: bar length carries the value and the axis carries the name, so colour names nothing.
    There is deliberately no pie chart: angles are hard to read, and ``normalize=True`` shows the
    same shares as bars.

    Args:
        data: ``{label: number}`` or a Series indexed by label (``value_counts()`` works).
        title: Figure title.
        subtitle: Muted line under the title.
        sort: ``True`` / ``"descending"`` largest first, ``"ascending"`` smallest first,
            ``False`` keeps the input order.
        top: Keep only the first ``top`` bars after sorting.
        horizontal: Horizontal bars with the first label at the top; ``False`` for vertical.
        normalize: Show each value as a share of the total (percent).
        error: Error bars: a ``{label: half-width}`` dict or Series for a symmetric bar, or a
            ``(minus, plus)`` pair of them for an asymmetric one (such as a confidence interval).
            Every bar needs an entry. The values are then written inside the bars so they do not
            collide with the whiskers.
        reference: Draw a line at this value.
        fmt: Python format spec for the written values (ignored when ``normalize``).
        xlabel: X axis title.
        ylabel: Y axis title.
        height: Figure height; grows with the number of bars by default.

    Raises:
        ConventionError: ``data`` is not a dict or Series (a DataFrame is refused).
        KeyError: ``error`` lacks a label that ``data`` has.
        ValueError: non-numeric values, negative errors, or ``normalize`` with a zero total.
    """
    s = as_value_series(data)
    total = float(s.sum())
    if normalize:
        if total == 0:
            raise ValueError("cannot normalise: the total is zero")
        s = s / total
    s = _ordered(s, sort, top)
    st = _style.get_style()
    minus = plus = None
    if error is not None:
        pair = error if isinstance(error, tuple) else (error, error)
        scale = total if normalize else 1.0
        minus, plus = ((as_value_series(e).reindex(s.index) / scale) for e in pair)
        if minus.isna().any() or plus.isna().any():
            missing = [str(k) for k in s.index[minus.isna() | plus.isna()]]
            raise KeyError(f"error has no entry for {missing}")
        if (minus < 0).any() or (plus < 0).any():
            raise ValueError("error bars cannot be negative")
    text = [format(v, ".1%" if normalize else fmt) for v in s]
    common = dict(
        marker={"color": _color.with_alpha(st.accent, 0.85), "line": {"width": 0}},
        text=text, textposition="inside" if minus is not None else "outside", cliponaxis=False,
        textfont={"size": 11, "color": st.ink_2}, showlegend=False,
    )  # fmt: skip
    if minus is not None and plus is not None:
        from .matrices import cell_ink

        common.update(insidetextanchor="start", textfont={"size": 11, "color": cell_ink(st.accent)})
        bar_error = {"type": "data", "symmetric": False, "array": plus.to_numpy(), "arrayminus": minus.to_numpy(),
                     "color": st.ink_2, "thickness": 1.5, "width": 5}  # fmt: skip
        common["error_x" if horizontal else "error_y"] = bar_error
    fig = go.Figure(go.Bar(x=s.to_numpy(), y=list(s.index), orientation="h", **common) if horizontal
                    else go.Bar(x=list(s.index), y=s.to_numpy(), **common))  # fmt: skip
    if reference is not None:
        fig.add_shape(type="line", line={"color": st.muted, "width": 1, "dash": "dot"},
                      **({"x0": reference, "x1": reference, "yref": "paper", "y0": 0, "y1": 1} if horizontal
                         else {"y0": reference, "y1": reference, "xref": "paper", "x0": 0, "x1": 1}))  # fmt: skip
    if horizontal:
        fig.update_yaxes(autorange="reversed", type="category", showgrid=False)
    else:
        fig.update_xaxes(type="category", showgrid=False)
    if normalize:
        (fig.update_xaxes if horizontal else fig.update_yaxes)(tickformat=".0%")
    return finish(fig, title, subtitle, xlabel=xlabel, ylabel=ylabel,
                  height=height or (_row_height(len(s)) if horizontal else HEIGHT), legend=False)  # fmt: skip


def plot_counts(
    values: pd.Series | Sequence[object],
    title: str = "",
    subtitle: str = "",
    *,
    normalize: bool = False,
    top: int | None = None,
    dropna: bool = True,
    **kwargs,
) -> go.Figure:
    """Count how often each value occurs and draw the counts as bars, in one call.

    Args:
        values: Raw observations (a Series or any sequence), one entry per occurrence.
        title: Figure title.
        subtitle: Muted line under the title.
        normalize: Show shares instead of counts.
        top: Keep only the ``top`` most frequent values.
        dropna: Leave missing values out (otherwise they are counted as ``NaN``).
        **kwargs: Passed to :func:`plot_bars` (``horizontal``, ``fmt``, ``height``, ...).
    """
    counts = pd.Series(list(values) if not isinstance(values, pd.Series) else values).value_counts(
        dropna=dropna
    )
    return plot_bars(counts, title, subtitle, normalize=normalize, top=top, **kwargs)


def plot_grouped_bars(
    frame: pd.DataFrame,
    title: str = "",
    subtitle: str = "",
    *,
    stacked: bool = False,
    normalize: bool = False,
    error: pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame] | None = None,
    horizontal: bool = False,
    xlabel: str = "",
    ylabel: str = "",
    height: int = HEIGHT,
) -> go.Figure:
    """Compare a few series across categories: index = categories, columns = series.

    Args:
        frame: Wide DataFrame (convention D). Each column becomes a coloured series.
        title: Figure title.
        subtitle: Muted line under the title.
        stacked: Stack the series instead of placing them side by side.
        normalize: Stack to 100% so each category shows shares (implies ``stacked``).
        error: Error bars: a DataFrame of half-widths with the same index and columns as ``frame``,
            or a ``(minus, plus)`` pair of them. Not available with ``normalize`` or ``stacked``,
            where a whisker would not mean one thing.
        horizontal: Horizontal bars with the first category at the top.
        xlabel: X axis title.
        ylabel: Y axis title.
        height: Figure height in px.

    Raises:
        ConventionError: ``frame`` is not a DataFrame.
        ValueError: more than ``MAX_SERIES`` columns, or negative values when stacking.
    """
    if not isinstance(frame, pd.DataFrame):
        raise convention_error("plot_grouped_bars", "D", frame)
    if frame.shape[1] > MAX_SERIES:
        raise ValueError(
            f"plot_grouped_bars draws at most {MAX_SERIES} series (one per colour), got {frame.shape[1]}"
        )
    stack = stacked or normalize
    if stack and (frame.to_numpy(dtype=float) < 0).any():
        raise ValueError("stacked bars need non-negative values")
    if error is not None and stack:
        raise ValueError("error bars are not available for stacked or normalised bars")
    minus = plus = None
    if error is not None:
        minus, plus = error if isinstance(error, tuple) else (error, error)
        for e in (minus, plus):
            if not (e.index.equals(frame.index) and e.columns.equals(frame.columns)):
                raise ValueError("error must have the same index and columns as frame")
            if (e.to_numpy(dtype=float) < 0).any():
                raise ValueError("error bars cannot be negative")
    cats = [str(i) for i in frame.index]
    shown = frame.div(frame.sum(axis=1).replace(0, np.nan), axis=0) if normalize else frame
    fig = go.Figure()
    for i, col in enumerate(frame.columns):
        v = shown[col].to_numpy(dtype=float)
        whisker: dict[str, object] = {}
        if minus is not None and plus is not None:
            whisker["error_x" if horizontal else "error_y"] = {
                "type": "data", "symmetric": False, "array": plus[col].to_numpy(dtype=float),
                "arrayminus": minus[col].to_numpy(dtype=float), "color": _style.get_style().ink_2,
                "thickness": 1.4, "width": 4,
            }  # fmt: skip
        fig.add_bar(name=str(col), marker={"color": _style.color(i), "line": {"width": 0}}, **whisker,
                    **({"x": v, "y": cats, "orientation": "h"} if horizontal else {"x": cats, "y": v}))  # fmt: skip
    fig.update_layout(barmode="relative" if stack else "group")
    if normalize:
        (fig.update_xaxes if horizontal else fig.update_yaxes)(tickformat=".0%")
    if horizontal:
        fig.update_yaxes(autorange="reversed", type="category")
    else:
        fig.update_xaxes(type="category")
    return finish(fig, title, subtitle, xlabel=xlabel, ylabel=ylabel, height=height, legend=True)


def plot_lollipop(
    data: Mapping[str, float] | pd.Series,
    title: str = "",
    subtitle: str = "",
    *,
    sort: Sort = True,
    top: int | None = None,
    baseline: float = 0.0,
    fmt: str = ",.4g",
    xlabel: str = "",
    height: int | None = None,
) -> go.Figure:
    """A stem and a dot per label: less ink than bars, clear for many categories or negatives.

    Args:
        data: ``{label: number}`` or a Series indexed by label.
        title: Figure title.
        subtitle: Muted line under the title.
        sort: As in :func:`plot_bars`.
        top: Keep only the first ``top`` after sorting.
        baseline: Where the stems start.
        fmt: Format spec for the written values.
        xlabel: X axis title.
        height: Figure height; grows with the number of rows.
    """
    s = _ordered(as_value_series(data), sort, top)
    st = _style.get_style()
    xs: list[float | None] = []
    ys: list[str | None] = []
    for label, v in s.items():
        xs += [baseline, float(v), None]
        ys += [str(label), str(label), None]
    fig = go.Figure()
    fig.add_scatter(
        x=xs,
        y=ys,
        mode="lines",
        line={"color": st.axis, "width": 2},
        hoverinfo="skip",
        showlegend=False,
    )
    fig.add_scatter(x=s.to_numpy(), y=list(s.index), mode="markers+text", showlegend=False,
                    marker={"color": st.accent, "size": 10}, text=[format(v, fmt) for v in s],
                    textposition="middle right", textfont={"size": 10, "color": st.ink_2})  # fmt: skip
    fig.update_yaxes(autorange="reversed", type="category", showgrid=False)
    return finish(
        fig, title, subtitle, xlabel=xlabel, height=height or _row_height(len(s)), legend=False
    )


def plot_dumbbell(
    frame: pd.DataFrame,
    label: str,
    start: str,
    end: str,
    title: str = "",
    subtitle: str = "",
    *,
    sort: bool = True,
    xlabel: str = "",
    height: int | None = None,
) -> go.Figure:
    """Before and after per item, joined by a line: how far did each one move?

    Args:
        frame: Long DataFrame, one row per item.
        label: Column naming the items.
        start: Numeric column with the first value (drawn in the muted tone).
        end: Numeric column with the second value (drawn in the accent).
        title: Figure title.
        subtitle: Muted line under the title.
        sort: Order items by the size of the move, largest first.
        xlabel: X axis title.
        height: Figure height; grows with the number of rows.
    """
    _require_columns(frame, "plot_dumbbell", label=label, start=start, end=end)
    names = _one_column(frame, label).astype(str).to_numpy()
    a, b = frame[start].to_numpy(dtype=float), frame[end].to_numpy(dtype=float)
    order = np.argsort(-np.abs(b - a), kind="stable") if sort else np.arange(len(frame))
    names, a, b = names[order], a[order], b[order]
    st = _style.get_style()
    xs: list[float | None] = []
    ys: list[str | None] = []
    for n, lo, hi in zip(names, a, b, strict=True):
        xs += [float(lo), float(hi), None]
        ys += [n, n, None]
    fig = go.Figure()
    fig.add_scatter(
        x=xs,
        y=ys,
        mode="lines",
        line={"color": st.axis, "width": 3},
        hoverinfo="skip",
        showlegend=False,
    )
    fig.add_scatter(
        x=a, y=names, mode="markers", name=start, marker={"color": st.muted, "size": 10}
    )
    fig.add_scatter(x=b, y=names, mode="markers", name=end, marker={"color": st.accent, "size": 10})
    fig.update_yaxes(autorange="reversed", type="category", showgrid=False)
    return finish(
        fig, title, subtitle, xlabel=xlabel, height=height or _row_height(len(names)), legend=True
    )


def plot_waterfall(
    data: Mapping[str, float] | pd.Series,
    title: str = "",
    subtitle: str = "",
    *,
    total: str | None = "Total",
    fmt: str = "+,.4g",
    ylabel: str = "",
    height: int = HEIGHT,
) -> go.Figure:
    """How contributions add up to a total, in the order given.

    Args:
        data: ``{label: contribution}`` in the order they apply (order is kept, not sorted).
        title: Figure title.
        subtitle: Muted line under the title.
        total: Label of the closing total bar; ``None`` omits it.
        fmt: Format spec for the written contributions.
        ylabel: Y axis title.
        height: Figure height in px.
    """
    s = as_value_series(data)
    st = _style.get_style()
    labels = list(s.index) + ([total] if total else [])
    measure = ["relative"] * len(s) + (["total"] if total else [])
    values = list(s.to_numpy()) + ([0.0] if total else [])
    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measure, connector={"line": {"color": st.axis, "width": 1}},
        increasing={"marker": {"color": st.categorical[0]}}, decreasing={"marker": {"color": st.categorical[1]}},
        totals={"marker": {"color": st.muted}},
        text=[format(v, fmt) for v in s] + ([format(float(s.sum()), ",.4g")] if total else []),
        textposition="outside", textfont={"size": 11, "color": st.ink_2}, showlegend=False,
    ))  # fmt: skip
    return finish(fig, title, subtitle, ylabel=ylabel, height=height, legend=False)


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

    Raises:
        KeyError: ``lower`` / ``upper`` lack a label present in ``data``.
        ValueError: more than ``MAX_SERIES`` series.
    """
    centre = pick(as_series_dict(data, _caller="plot_line_band"), series)
    lo = as_series_dict(lower, _caller="plot_line_band")
    hi = as_series_dict(upper, _caller="plot_line_band")
    if len(centre) > MAX_SERIES:
        raise ValueError(f"plot_line_band draws at most {MAX_SERIES} series, got {len(centre)}")
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
        fig, title, subtitle, xlabel=xlabel, ylabel=ylabel, height=height, legend=len(centre) > 1
    )


def _ols(x: np.ndarray, y: np.ndarray, degree: int, grid: np.ndarray, z: float = 1.96):
    """Polynomial least squares with a normal-approximation band for the mean prediction."""
    design = np.vander(x, degree + 1)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = max(len(x) - degree - 1, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    g = np.vander(grid, degree + 1)
    mean = g @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", g, cov, g))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return mean, mean - z * se, mean + z * se, r2, beta


def plot_regression(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    subtitle: str = "",
    *,
    group: str | None = None,
    degree: int = 1,
    band: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    height: int = HEIGHT,
) -> go.Figure:
    """Scatter with an ordinary-least-squares fit and a 95% band for the mean (numpy only).

    The band is a normal approximation for the *mean* prediction, not a prediction interval for
    new observations.

    Args:
        frame: Long DataFrame.
        x: Numeric column for x.
        y: Numeric column for y.
        title: Figure title.
        subtitle: Muted line under the title; a default reports the slope and R² when ``degree=1``.
        group: Column splitting the data into separately fitted groups (at most 3 values).
        degree: Polynomial degree of the fit.
        band: Draw the confidence band.
        xlabel: X axis title; defaults to the column name.
        ylabel: Y axis title; defaults to the column name.
        height: Figure height in px.

    Raises:
        ValueError: too few points for the degree, or more than 3 groups.
    """
    _require_columns(frame, "plot_regression", x=x, y=y, group=group)
    gcol = _one_column(frame, group).astype(str).to_numpy() if group else np.full(len(frame), "")
    names = list(pd.unique(gcol))
    if len(names) > 3:
        raise ValueError(
            f"plot_regression: group={group!r} has {len(names)} values; at most 3 (read pair by pair)"
        )
    fig = go.Figure()
    notes = []
    for i, name in enumerate(names):
        mask = gcol == name
        xs = frame[x].to_numpy(dtype=float)[mask]
        ys = frame[y].to_numpy(dtype=float)[mask]
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        if len(xs) <= degree + 1:
            raise ValueError(
                f"plot_regression: need more than {degree + 1} points for degree {degree}, got {len(xs)}"
            )
        color = _style.color(i)
        grid = np.linspace(xs.min(), xs.max(), 100)
        mean, lo, hi, r2, beta = _ols(xs, ys, degree, grid)
        if band:
            fig.add_scatter(x=np.r_[grid, grid[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", mode="lines", line={"width": 0},
                            fillcolor=_color.with_alpha(color, 0.16), hoverinfo="skip", showlegend=False,
                            legendgroup=name)  # fmt: skip
        fig.add_scatter(x=xs, y=ys, mode="markers", name=name or "observations", legendgroup=name,
                        marker={**_style.marker(i, size=7), "opacity": 0.75}, showlegend=bool(group))  # fmt: skip
        fig.add_scatter(x=grid, y=mean, mode="lines", showlegend=False, legendgroup=name, hoverinfo="skip",
                        line={"color": color, "width": 2.2})  # fmt: skip
        notes.append(
            f"{name + ': ' if name else ''}slope {beta[-2]:.3g}, R² {r2:.2f}"
            if degree == 1
            else f"{name + ': ' if name else ''}R² {r2:.2f}"
        )
    return finish(fig, title, subtitle or "; ".join(notes), xlabel=xlabel or x, ylabel=ylabel or y, height=height,
                  legend=bool(group))  # fmt: skip


def plot_scatter_matrix(
    frame: pd.DataFrame,
    title: str = "",
    subtitle: str = "",
    *,
    columns: Sequence[str] | None = None,
    bins: int = 30,
    cell: int = 150,
) -> go.Figure:
    """Pairwise scatter of up to 6 numeric columns with histograms on the diagonal.

    Args:
        frame: DataFrame with numeric columns.
        title: Figure title.
        subtitle: Muted line under the title.
        columns: Columns to include (default: all numeric ones).
        bins: Bins of the diagonal histograms.
        cell: Size of one cell in px.

    Raises:
        ValueError: fewer than 2 or more than 6 columns.
    """
    if not isinstance(frame, pd.DataFrame):
        raise convention_error("plot_scatter_matrix", "D", frame)
    cols = list(columns) if columns is not None else list(frame.select_dtypes("number").columns)
    if not 2 <= len(cols) <= 6:
        raise ValueError(f"plot_scatter_matrix needs 2 to 6 columns, got {len(cols)}")
    n = len(cols)
    st = _style.get_style()
    fig = make_subplots(
        rows=n,
        cols=n,
        shared_xaxes="columns",  # type: ignore[arg-type], horizontal_spacing=0.012, vertical_spacing=0.012
    )
    for r, ry in enumerate(cols, start=1):
        for c, cx in enumerate(cols, start=1):
            if r == c:
                counts, edges = np.histogram(frame[cx].dropna(), bins=bins)
                fig.add_bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, marker={"color": _color.with_alpha(st.accent, 0.7), "line": {"width": 0}},
                            showlegend=False, hoverinfo="skip", row=r, col=c)  # fmt: skip
            elif r > c:
                fig.add_scatter(x=frame[cx], y=frame[ry], mode="markers", showlegend=False,
                                marker={"color": _color.with_alpha(st.accent, 0.5), "size": 4}, row=r, col=c)  # fmt: skip
            if c == 1:
                fig.update_yaxes(title_text=ry, row=r, col=c)
            if r == n:
                fig.update_xaxes(title_text=cx, row=r, col=c)
    fig.update_layout(bargap=0)
    return finish(
        fig, title, subtitle, height=max(HEIGHT, 110 + cell * n), width=None, legend=False
    )


def plot_calendar_heatmap(
    data: pd.Series,
    title: str = "",
    subtitle: str = "",
    *,
    fmt: str = ".4g",
    height: int = 260,
) -> go.Figure:
    """A daily series as a weekday-by-week grid, shaded by value (like a contribution calendar).

    Args:
        data: Series with a DatetimeIndex; several values on one day are summed.
        title: Figure title.
        subtitle: Muted line under the title.
        fmt: Format spec for the hover value.
        height: Figure height in px.

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
    return finish(fig, title, subtitle, height=height, legend=False)


def plot_pareto(
    data: Mapping[str, float] | pd.Series,
    title: str = "",
    subtitle: str = "",
    *,
    top: int | None = None,
    other: str = "Other",
    threshold: float | None = 0.8,
    xlabel: str = "",
    height: int = HEIGHT,
) -> go.Figure:
    """A Pareto chart: causes ranked by size with the running share of the total.

    Bars are each cause's share of the total, largest first, and the line is the cumulative share,
    so you can see how few causes account for most of the total. Both are shares, so they share one
    axis (the classic two-axis version is deliberately not offered).

    Args:
        data: ``{label: non-negative number}`` or a Series indexed by label (``value_counts()``).
        title: Figure title.
        subtitle: Muted line under the title.
        top: Show this many causes and fold the rest into one bar named ``other``.
        other: Label of the folded bar.
        threshold: Draw a dotted line at this cumulative share (80% for the "80/20" reading);
            ``None`` for none.
        xlabel: X axis title.
        height: Figure height in px.

    Raises:
        ConventionError: ``data`` is not a dict or Series.
        ValueError: negative values or a zero total.
    """
    s = as_value_series(data)
    if (s < 0).any():
        raise ValueError("plot_pareto needs non-negative values")
    total = float(s.sum())
    if total == 0:
        raise ValueError("cannot rank: the total is zero")
    s = s.sort_values(ascending=False, kind="stable")
    if top is not None and len(s) > top:
        s = pd.concat([s.head(top), pd.Series({other: float(s.iloc[top:].sum())})])
    shares = s / total
    st = _style.get_style()
    labels = list(shares.index)
    fig = go.Figure(go.Bar(x=labels, y=shares.to_numpy(), name="share", showlegend=False,
                           marker={"color": _color.with_alpha(st.accent, 0.85), "line": {"width": 0}},
                           text=[format(v, ".0%") for v in shares], textposition="outside", cliponaxis=False,
                           textfont={"size": 10, "color": st.ink_2}))  # fmt: skip
    fig.add_scatter(x=labels, y=np.cumsum(shares.to_numpy()), mode="lines+markers", name="cumulative share",
                    line={"color": _style.color(1), "width": 2}, marker=_style.marker(1, size=6))  # fmt: skip
    if threshold is not None:
        fig.add_hline(y=threshold, line={"color": st.muted, "width": 1, "dash": "dot"})
    fig.update_xaxes(type="category", showgrid=False)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.08])
    return finish(
        fig, title, subtitle, xlabel=xlabel, ylabel="share of total", height=height, legend=True
    )
