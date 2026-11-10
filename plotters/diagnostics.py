"""Time-series diagnostics: where the data is missing, seasonality, lags, cross-correlation, spectrum.

These look at a series (or two) before or beside modelling: is the data complete, is there a
repeating pattern, does today resemble yesterday, does one series lead another.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import NormalDist
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import _color
from . import style as _style
from ._errors import convention_error
from ._types import SeriesLike
from .correlogram import _add_panel
from .frames import as_series_dict, pick
from .layout import HEIGHT, apply_log, finish
from .stats import ccf_values, periodogram

__all__ = ["plot_gaps", "plot_seasonal", "plot_lag", "plot_ccf", "plot_periodogram"]


def plot_gaps(
    data: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    series: Sequence[str] | None = None,
    bins: int = 120,
    sort: bool = False,
    height: int | None = None,
    width: int | None = None,
) -> go.Figure:
    """Where each series has missing data: one row per series, time along the x axis.

    A row is cut into ``bins`` stretches of consecutive rows and each stretch is shaded by the share
    of values missing in it, so a long outage and a scatter of single gaps look different. The
    series are lined up on the union of their index, so a date a series simply does not have counts
    as missing; non-finite values (``inf``) count as missing too. Each row label carries the
    overall share missing.

    Args:
        data: Series, wide DataFrame or ``{label: Series}``.
        title: Figure title.
        subtitle: Muted line under the title.
        series: Subset and order of series.
        bins: Number of stretches along the time axis (fewer if there are fewer rows).
        sort: Order rows by the share missing, worst first, instead of the input order.
        height: Figure height; grows with the number of series.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ValueError: no data.
    """
    frames = pick(as_series_dict(data, _caller="plot_gaps"), series)
    wide = pd.concat({str(k): v for k, v in frames.items()}, axis=1).sort_index()
    if wide.empty:
        raise ValueError("plot_gaps needs some data")
    numeric = wide.apply(pd.to_numeric, errors="coerce")
    missing = (~np.isfinite(numeric.to_numpy(dtype=float))).T  # rows = series
    names = list(wide.columns)
    overall = missing.mean(axis=1)
    order = np.argsort(-overall, kind="stable") if sort else np.arange(len(names))
    chunks = np.array_split(np.arange(wide.shape[0]), min(bins, wide.shape[0]))
    z = np.array([[missing[i, c].mean() for c in chunks] for i in order])
    index = wide.index
    starts = [index[c[0]] for c in chunks]
    spans = [[f"{index[c[0]]} to {index[c[-1]]}" for c in chunks] for _ in order]
    labels = [f"{names[i]} ({overall[i]:.0%} missing)" for i in order]
    st = _style.get_style()
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(s) if not isinstance(s, pd.Timestamp) else s for s in starts], y=labels, zmin=0, zmax=1,
        colorscale=_style.sequential_scale(), xgap=0, ygap=2, customdata=spans,
        hovertemplate="%{y}<br>%{customdata}<br>%{z:.0%} missing<extra></extra>",
        colorbar={"title": {"text": "missing", "font": {"size": 11, "color": st.ink_2}}, "thickness": 10,
                  "outlinewidth": 0, "tickformat": ".0%", "tickfont": {"size": 10, "color": st.muted}},
    ))  # fmt: skip
    fig.update_yaxes(autorange="reversed", type="category", showgrid=False)
    fig.update_xaxes(showgrid=False)
    widest = max(len(label) for label in labels)
    fig = finish(
        fig,
        title,
        subtitle or "colour = the share of each stretch that is missing",
        height=height or int(min(900, max(240, 140 + 26 * len(names)))),
        legend=False,
        width=width,
    )
    fig.update_layout(
        margin={"l": int(min(320, max(60, widest * 6.6 + 40)))}
    )  # row labels carry the % missing
    return fig


_PERIODS: dict[str, tuple[str, list[str]]] = {
    "month": (
        "year",
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    ),
    "quarter": ("year", ["Q1", "Q2", "Q3", "Q4"]),
    "weekday": ("ISO week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
    "hour": ("day", [f"{h:02d}h" for h in range(24)]),
}


def plot_seasonal(
    series: pd.Series,
    title: str = "",
    subtitle: str = "",
    *,
    period: Literal["month", "quarter", "weekday", "hour"] = "month",
    agg: Literal["mean", "median", "sum", "last"] = "mean",
    max_cycles: int = 12,
    show_average: bool = True,
    ylabel: str = "",
    height: int = HEIGHT,
    width: int | None = None,
) -> go.Figure:
    """One line per cycle laid over the same axis, so a repeating pattern shows up as parallel lines.

    With ``period="month"`` each year is a line running January to December; a seasonal series
    draws the same shape every year, and a level shift shows as lines stacked above one another.
    Older cycles are drawn fainter and recent ones stronger. A bold dashed line is the average.

    Args:
        series: A Series with a DatetimeIndex (several observations in one period are combined with
            ``agg``).
        title: Figure title.
        subtitle: Muted line under the title.
        period: The repeating unit on the x axis: ``"month"`` and ``"quarter"`` (one line per year),
            ``"weekday"`` (one line per ISO week) or ``"hour"`` (one line per day).
        agg: How to combine observations falling in the same cycle and position.
        max_cycles: Show only the most recent cycles (a chart of 40 lines says nothing).
        show_average: Draw the average across cycles.
        ylabel: Y axis title.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ValueError: the index is not a DatetimeIndex, or an unknown ``period`` / ``agg``.
    """
    if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("plot_seasonal needs a Series with a DatetimeIndex")
    if period not in _PERIODS:
        raise ValueError(f"period must be one of {list(_PERIODS)}, got {period!r}")
    if agg not in ("mean", "median", "sum", "last"):
        raise ValueError(f"agg must be 'mean', 'median', 'sum' or 'last', got {agg!r}")
    idx = series.index
    cycle_name, ticks = _PERIODS[period]
    if period == "month":
        cycle, pos = idx.year, idx.month - 1
    elif period == "quarter":
        cycle, pos = idx.year, idx.quarter - 1
    elif period == "weekday":
        iso = idx.isocalendar()
        cycle, pos = (
            (iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)).to_numpy(),
            idx.day_of_week,
        )
    else:
        cycle, pos = idx.strftime("%Y-%m-%d"), idx.hour
    frame = pd.DataFrame(
        {
            "cycle": np.asarray(cycle),
            "pos": np.asarray(pos),
            "v": pd.to_numeric(series, errors="coerce").to_numpy(),
        }
    )
    table = (
        frame.groupby(["cycle", "pos"])["v"].agg(agg).unstack("cycle").reindex(range(len(ticks)))
    )
    table = table[sorted(table.columns)].iloc[:, -max_cycles:]
    st = _style.get_style()
    fig = go.Figure()
    for k, col in enumerate(table.columns):
        t = 0.3 + 0.7 * (k / max(len(table.columns) - 1, 1))
        fig.add_scatter(x=list(range(len(ticks))), y=table[col].to_numpy(), mode="lines+markers", name=str(col),
                        line={"color": _ramp(st.sequential, t), "width": 1.4 + 1.2 * (k == len(table.columns) - 1)},
                        marker={"size": 4, "color": _ramp(st.sequential, t)}, connectgaps=False)  # fmt: skip
    if show_average and len(table.columns) > 1:
        fig.add_scatter(x=list(range(len(ticks))), y=table.mean(axis=1).to_numpy(), mode="lines", name="average",
                        line={"color": st.ink, "width": 2.6, "dash": "dash"})  # fmt: skip
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(ticks))), ticktext=ticks)
    return finish(fig, title, subtitle or f"one line per {cycle_name}, older = fainter", ylabel=ylabel,
                  height=height, legend=len(table.columns) <= 14, width=width)  # fmt: skip


def _ramp(ramp: tuple[str, ...], t: float) -> str:
    from .matrices import _ramp_color

    return _ramp_color(ramp, t)


def plot_lag(
    series: pd.Series,
    title: str = "",
    subtitle: str = "",
    *,
    lags: int | Sequence[int] = 1,
    ncols: int = 3,
    panel_height: int = 300,
    height: int | None = None,
    width: int | None = None,
) -> go.Figure:
    """Each observation against the one ``k`` steps earlier: does today look like yesterday?

    A cloud around the diagonal means persistence, a shapeless blob means none, and structure
    (curves, clusters) is something a single correlation coefficient would hide. Each panel title
    carries the correlation.

    Args:
        series: Observations in order.
        title: Figure title.
        subtitle: Muted line under the title.
        lags: One lag or several; one panel each.
        ncols: Panels per row.
        panel_height: Height of one panel row in px.
        height: Total figure height in px; overrides the height computed from ``panel_height``.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ConventionError: ``series`` is not a Series.
        ValueError: a lag below 1 or not smaller than the series length.
    """
    if not isinstance(series, pd.Series):
        raise convention_error("plot_lag", "C", series, "Pass one Series.")
    ks = [lags] if isinstance(lags, int) else list(lags)
    x = pd.to_numeric(series, errors="coerce").astype(float)
    if any(k < 1 or k >= len(x) for k in ks):
        raise ValueError(f"lags must be between 1 and {len(x) - 1}")
    st = _style.get_style()
    rows = -(-len(ks) // ncols)
    pairs = [pd.concat([x.shift(k), x], axis=1).dropna() for k in ks]
    titles = [
        f"lag {k} (r = {p.iloc[:, 0].corr(p.iloc[:, 1]):.2f})"
        for k, p in zip(ks, pairs, strict=True)
    ]
    fig = make_subplots(rows=rows, cols=min(ncols, len(ks)), subplot_titles=titles, horizontal_spacing=0.08,
                        vertical_spacing=min(0.16, 0.7 / rows))  # fmt: skip
    for n, p in enumerate(pairs):
        r, c = n // ncols + 1, n % ncols + 1
        fig.add_scatter(x=p.iloc[:, 0], y=p.iloc[:, 1], mode="markers", showlegend=False,
                        marker={"size": 4, "color": st.accent, "opacity": 0.55}, row=r, col=c)  # fmt: skip
        lo, hi = float(min(p.min().min(), x.min())), float(max(p.max().max(), x.max()))
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False, hoverinfo="skip",
                        line={"color": st.muted, "width": 1, "dash": "dot"}, row=r, col=c)  # fmt: skip
        fig.update_xaxes(title_text=f"x[t-{ks[n]}]", row=r, col=c)
    fig.update_yaxes(title_text="x[t]", col=1)
    fig.update_annotations(font={"size": 11, "color": st.ink_2})
    return finish(
        fig,
        title,
        subtitle,
        height=height or max(HEIGHT, 120 + panel_height * rows),
        legend=False,
        width=width,
    )


def plot_ccf(
    x: pd.Series,
    y: pd.Series,
    title: str = "",
    subtitle: str = "",
    *,
    lags: int = 20,
    alpha: float = 0.05,
    height: int = HEIGHT,
    width: int | None = None,
) -> go.Figure:
    """Cross-correlation of two series at lags in both directions: does one lead the other?

    The value at lag ``k`` is the correlation of ``x[t]`` with ``y[t - k]``. A peak at a **positive**
    lag means ``x`` follows what ``y`` did ``k`` steps earlier; a peak at a **negative** lag means ``y``
    follows ``x``. The ribbon is the flat ``z / sqrt(n)`` band for two unrelated white-noise series;
    trending or autocorrelated series produce spurious peaks, so difference or pre-whiten them first.

    Args:
        x: First series.
        y: Second series. Aligned with ``x`` on the index when both have one.
        title: Figure title.
        subtitle: Muted line under the title; a default states which direction is which.
        lags: Highest lag in each direction.
        alpha: Two-sided level of the band.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ConventionError: ``x`` or ``y`` is not a Series.
        ValueError: ``lags`` too large for the sample, or a constant series.
    """
    for name, s in (("x", x), ("y", y)):
        if not isinstance(s, pd.Series):
            raise convention_error("plot_ccf", "C", s, f"{name} must be a Series.")
    lag, coef = ccf_values(x, y, lags)
    n = int(pd.concat([x, y], axis=1, join="inner").dropna().shape[0])
    z = NormalDist().inv_cdf(1 - alpha / 2)
    band = np.full(coef.shape, z / np.sqrt(n))
    xn, yn = str(x.name or "x"), str(y.name or "y")
    fig = go.Figure()
    _add_panel(fig, lag.astype(float), coef, band, _style.get_style().accent, "", legend=True)
    fig.add_hline(y=0, line={"color": _style.get_style().axis, "width": 1})
    default = f"positive lag: {xn} follows {yn}; negative lag: {yn} follows {xn}"
    return finish(
        fig,
        title,
        subtitle or default,
        xlabel="lag",
        ylabel=f"corr({xn}[t], {yn}[t-k])",
        height=height,
        legend=True,
        width=width,
    )


def plot_periodogram(
    series: pd.Series | np.ndarray | Sequence[float],
    title: str = "",
    subtitle: str = "",
    *,
    x: Literal["frequency", "period"] = "frequency",
    spacing: float = 1.0,
    detrend: bool = True,
    peaks: int = 3,
    logy: bool = True,
    height: int = HEIGHT,
    width: int | None = None,
) -> go.Figure:
    """Where the variance of a series sits across frequencies: a cycle shows as a spike.

    The raw periodogram is noisy (each point has a wide sampling error that does not shrink with
    more data), so read the large, isolated spikes, not the ripple. The largest ones are labelled
    with their period.

    Args:
        series: Observations at equal spacing.
        title: Figure title.
        subtitle: Muted line under the title.
        x: ``"frequency"`` (cycles per unit of time) or ``"period"`` (its reciprocal, on a log axis).
        spacing: Time between observations, to give periods and frequencies in your own units
            (12 monthly observations per year: ``spacing=1/12`` gives cycles per year).
        detrend: Remove a straight-line trend first so it does not swamp the low frequencies.
        peaks: How many of the largest spikes to label.
        logy: Logarithmic power axis (power spans decades).
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ValueError: fewer than 8 observations or an unknown ``x``.
    """
    if x not in ("frequency", "period"):
        raise ValueError(f"x must be 'frequency' or 'period', got {x!r}")
    freq, power = periodogram(series, detrend=detrend, spacing=spacing)
    st = _style.get_style()
    xs = freq if x == "frequency" else 1.0 / freq
    order = np.argsort(xs)
    fig = go.Figure(go.Scatter(x=xs[order], y=power[order], mode="lines", line={"color": st.accent, "width": 1.6},
                               fill="tozeroy", fillcolor=_color.with_alpha(st.accent, 0.12),
                               hovertemplate=f"{x} %{{x:.4g}}: %{{y:.3g}}<extra></extra>", showlegend=False))  # fmt: skip
    chosen: list[int] = []
    for j in np.argsort(-power):
        if all(abs(int(j) - c) > 2 for c in chosen):
            chosen.append(int(j))
        if len(chosen) >= peaks:
            break
    for j in chosen:
        fig.add_annotation(x=xs[j], y=float(np.log10(power[j])) if logy else power[j], text=f"period {1 / freq[j]:.3g}",
                           showarrow=True, arrowhead=0, ax=18, ay=-22, arrowcolor=st.muted,
                           font={"size": 10, "color": st.ink_2})  # fmt: skip
    if x == "period":
        apply_log(fig, "x", float(xs.min()), float(xs.max()))
    if logy:
        positive = power[power > 0]
        apply_log(fig, "y", float(positive.min()) / 1.5, float(positive.max()) * 1.6)
    return finish(
        fig,
        title,
        subtitle,
        xlabel="period" if x == "period" else "frequency (cycles per unit)",
        ylabel="power",
        height=height,
        legend=False,
        width=width,
    )
