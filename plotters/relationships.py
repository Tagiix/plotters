"""Relationships between numeric columns: scatter, bubbles, density, regression and pairwise plots.

Convention B (a long frame plus column-name strings) except :func:`plot_scatter_matrix`, which takes
a wide frame.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import _color
from . import style as _style
from ._checks import _MARKER_SIZE, _cap_groups, _log_bounds, _numeric
from ._errors import convention_error
from .frames import _one_column, _require_columns
from .layout import HEIGHT, apply_log, finish

_MAX_WRITTEN_LABELS = 40
"""Past this many points a label is hover-only: written next to every point it is unreadable."""

__all__ = [
    "plot_scatter",
    "plot_shaded_scatter",
    "plot_density2d",
    "plot_regression",
    "plot_scatter_matrix",
]


def _bubble_sizes(
    values: np.ndarray, caller: str, column: str, largest: float = 30.0
) -> np.ndarray:
    """Marker diameters in px so that the *area* is proportional to the value.

    Area, not diameter, is what the eye reads: doubling a value should double the area. The
    smallest visible marker is 4 px, so a tiny value does not vanish.
    """
    finite = values[np.isfinite(values)]
    if finite.size and (finite < 0).any():
        raise ValueError(f"{caller}: size={column!r} has negative values; area cannot be negative")
    top = float(finite.max()) if finite.size else 0.0
    if top <= 0:
        return np.full(values.shape, 4.0)
    return np.maximum(4.0, largest * np.sqrt(np.nan_to_num(values) / top))


def plot_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    subtitle: str = "",
    *,
    group: str | None = None,
    label: str | None = None,
    size: str | None = None,
    diagonal: bool = False,
    logx: bool = False,
    logy: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    height: int = HEIGHT,
    width: int | None = None,
) -> go.Figure:
    """Two columns against each other, optionally split by group, annotated by label, or sized.

    ``group=`` and ``label=`` are different things and easy to swap::

        plot_scatter(long, x="cost", y="error", group="model")  # one coloured series per model
        plot_scatter(long, x="cost", y="error", label="model")  # one series, text next to points

    Args:
        frame: Long DataFrame, one row per observation.
        x: Column for the x axis.
        y: Column for the y axis.
        title: Figure title.
        subtitle: Muted line under the title.
        group: Column whose values each become one coloured series (at most 3 values).
        label: Column naming each point. Always shown on hover; also written next to the points
            when there are at most 40 of them (more would overprint).
        size: Non-negative numeric column that sets each marker's **area** (a bubble chart). The
            default subtitle says so.
        diagonal: Draw ``y = x``.
        logx: Logarithmic x axis (needs positive values).
        logy: Logarithmic y axis.
        xlabel: X axis title; defaults to the column name.
        ylabel: Y axis title; defaults to the column name.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ConventionError: ``frame`` is not a DataFrame (a Series or dict was passed).
        KeyError: a named column is missing.
        ValueError: more than 3 groups, a non-numeric axis column, or a log axis with
            non-positive values.
    """
    _require_columns(frame, "plot_scatter", x=x, y=y, group=group, label=label, size=size)
    xs, ys = _numeric(frame, x, "plot_scatter"), _numeric(frame, y, "plot_scatter")
    sizes = (
        _bubble_sizes(_numeric(frame, size, "plot_scatter"), "plot_scatter", size) if size else None
    )
    st = _style.get_style()
    fig = go.Figure()
    parts: list[tuple[str, pd.Index]] = [("", frame.index)]
    if group is not None:
        values = list(pd.unique(_one_column(frame, group)))
        _cap_groups(values, "plot_scatter", group)
        parts = [(str(v), frame.index[_one_column(frame, group).to_numpy() == v]) for v in values]
    for i, (name, idx) in enumerate(parts):
        mask = frame.index.isin(idx)
        text = _one_column(frame, label).astype(str).to_numpy()[mask] if label else None
        marker = {
            **_style.marker(i, size=_MARKER_SIZE),
            "color": _style.color(i) if group else st.accent,
        }
        if sizes is not None:
            marker.update(
                size=sizes[mask],
                sizemode="diameter",
                opacity=0.7,
                line={"width": 1, "color": st.surface},
            )
        written = bool(label) and int(mask.sum()) <= _MAX_WRITTEN_LABELS
        fig.add_scatter(
            x=xs[mask], y=ys[mask], mode="markers+text" if written else "markers", name=name,
            marker=marker, text=text, textposition="top right", textfont={"size": 10, "color": st.muted},
            hovertemplate=(f"%{{text}}<br>{x}=%{{x:.4g}}<br>{y}=%{{y:.4g}}<extra>{name}</extra>" if label else None),
            showlegend=group is not None,
        )  # fmt: skip
    fig.update_layout(hovermode="closest")  # the theme's "x unified" cannot pick one point
    if diagonal:
        lo, hi = float(min(np.nanmin(xs), np.nanmin(ys))), float(max(np.nanmax(xs), np.nanmax(ys)))
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x", showlegend=False,
                        line={"color": st.muted, "width": 1, "dash": "dot"}, hoverinfo="skip")  # fmt: skip
    if logx:
        apply_log(fig, "x", *_log_bounds(xs, "plot_scatter", "x"))
    if logy:
        apply_log(fig, "y", *_log_bounds(ys, "plot_scatter", "y"))
    note = f"marker area is proportional to {size}" if size else ""
    return finish(fig, title, subtitle or note, xlabel=xlabel or x, ylabel=ylabel or y, height=height,
                  legend=group is not None, width=width)  # fmt: skip


def plot_shaded_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    shade: str,
    title: str = "",
    subtitle: str = "",
    *,
    label: str | None = None,
    size: str | None = None,
    log_shade: bool = False,
    reference: float | None = None,
    clip: float = 0.01,
    logx: bool = False,
    logy: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    height: int = HEIGHT,
    width: int | None = None,
) -> go.Figure:
    """Two columns plus a **continuous** third as colour, with a colourbar.

    Use this instead of ``plot_scatter(group=)`` when the third variable is a quantity rather
    than a category: ``group=`` caps at three hues and throws the gradient away.

    Args:
        frame: Long DataFrame.
        x: Column for the x axis.
        y: Column for the y axis.
        shade: Numeric column that sets the colour.
        title: Figure title.
        subtitle: Muted line under the title.
        label: Column naming each point. Always shown on hover; also written next to the points
            when there are at most 40 of them (more would overprint).
        size: Non-negative numeric column that sets each marker's area, so one chart carries four
            variables (x, y, colour and size).
        log_shade: Colour by ``log10(shade)`` for a geometric ratio (needs positive values);
            the colourbar is still labelled in data units.
        reference: A shade value that is "neutral" (such as 1.0 for a ratio, 0 for a gain). Uses
            the diverging ramp centred on it, symmetric in log space when ``log_shade``, but only
            when the data lie on both sides of it; otherwise (all values above, or a reference
            that is not positive under ``log_shade``) the sequential ramp is used, since a
            diverging ramp would paint every point the same colour.
        clip: Fraction of the shade values ignored at each end when setting the colour range, so
            a few outliers do not squash the gradient for everyone else (they take the end
            colour). ``0`` uses the full range.
        logx: Logarithmic x axis.
        logy: Logarithmic y axis.
        xlabel: X axis title; defaults to the column name.
        ylabel: Y axis title; defaults to the column name.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ValueError: ``shade`` is not numeric, ``log_shade`` with non-positive values, or ``clip``
            outside [0, 0.5).
    """
    caller = "plot_shaded_scatter"
    _require_columns(frame, caller, x=x, y=y, shade=shade, label=label, size=size)
    xs, ys, sh = (
        _numeric(frame, x, caller),
        _numeric(frame, y, caller),
        _numeric(frame, shade, caller),
    )
    if not 0 <= clip < 0.5:
        raise ValueError(f"{caller}: clip must be in [0, 0.5), got {clip}")
    st = _style.get_style()
    values = sh
    if log_shade:
        if (sh[np.isfinite(sh)] <= 0).any():
            raise ValueError(f"{caller}: log_shade needs strictly positive shade values")
        values = np.log10(sh)
    finite = values[np.isfinite(values)]
    cmin, cmax = (float(v) for v in np.quantile(finite, [clip, 1 - clip], method="inverted_cdf"))
    marker: dict[str, object] = {
        "size": _bubble_sizes(_numeric(frame, size, caller), caller, size)
        if size
        else _MARKER_SIZE + 1,
        "color": values,
        "line": {"width": 0.6, "color": st.surface},
    }
    center = None
    if reference is not None and (reference > 0 or not log_shade):
        center = math.log10(reference) if log_shade else reference
    if center is not None and cmin < center < cmax:
        span = max(center - cmin, cmax - center)
        marker.update(colorscale=_style.diverging_scale(), cmin=center - span, cmax=center + span)
    else:  # no neutral point inside the data: a diverging ramp would paint everything one colour
        marker.update(
            colorscale=_style.sequential_scale(), cmin=cmin, cmax=cmax if cmax > cmin else cmin + 1
        )
    ticks = np.linspace(marker["cmin"], marker["cmax"], 5)  # type: ignore[arg-type]
    marker["colorbar"] = {
        "title": {"text": shade, "font": {"size": 11, "color": st.ink_2}}, "thickness": 12, "outlinewidth": 0,
        "tickfont": {"size": 10, "color": st.muted}, "tickmode": "array", "tickvals": ticks.tolist(),
        "ticktext": [f"{10**t:.3g}" if log_shade else f"{t:.3g}" for t in ticks],
    }  # fmt: skip
    text = _one_column(frame, label).astype(str).to_numpy() if label else None
    written = bool(label) and len(xs) <= _MAX_WRITTEN_LABELS
    head = "%{text}<br>" if label else ""
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers+text" if written else "markers", marker=marker, text=text,
        textposition="top right", textfont={"size": 10, "color": st.muted}, customdata=sh,
        hovertemplate=f"{head}{x}=%{{x:.4g}}<br>{y}=%{{y:.4g}}<br>{shade}=%{{customdata:.4g}}<extra></extra>",
    ))  # fmt: skip
    fig.update_layout(hovermode="closest")
    if logx:
        apply_log(fig, "x", *_log_bounds(xs, caller, "x"))
    if logy:
        apply_log(fig, "y", *_log_bounds(ys, caller, "y"))
    return finish(
        fig,
        title,
        subtitle,
        xlabel=xlabel or x,
        ylabel=ylabel or y,
        height=height,
        legend=False,
        width=width,
    )


def plot_density2d(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    subtitle: str = "",
    *,
    bins: int = 60,
    log_counts: bool = True,
    logx: bool = False,
    logy: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    height: int = HEIGHT,
    width: int | None = None,
) -> go.Figure:
    """Two columns as a binned density: the scatter that stays readable with many points.

    Past a few thousand points a scatter overplots into a solid blob and hides where the mass is.
    Here the plane is cut into cells and each is shaded by how many points it holds; empty cells
    stay blank.

    Args:
        frame: Long DataFrame.
        x: Column for the x axis.
        y: Column for the y axis.
        title: Figure title.
        subtitle: Muted line under the title; a default reports the number of points.
        bins: Cells per axis.
        log_counts: Shade by ``log10(count)`` so sparse tails stay visible next to a dense core;
            the colourbar is still labelled in counts.
        logx: Logarithmic x axis; cells are equal-width in log space (needs positive values).
        logy: Logarithmic y axis.
        xlabel: X axis title; defaults to the column name.
        ylabel: Y axis title; defaults to the column name.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ValueError: no finite pairs, or a log axis with non-positive values.
    """
    caller = "plot_density2d"
    _require_columns(frame, caller, x=x, y=y)
    xs, ys = _numeric(frame, x, caller), _numeric(frame, y, caller)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if xs.size == 0:
        raise ValueError(f"{caller}: no finite (x, y) pairs")
    if logx:
        lo, hi = _log_bounds(xs, caller, "x")
    if logy:
        lo_y, hi_y = _log_bounds(ys, caller, "y")
    xe = (
        np.geomspace(xs.min(), xs.max(), bins + 1)
        if logx
        else np.linspace(xs.min(), xs.max(), bins + 1)
    )
    ye = (
        np.geomspace(ys.min(), ys.max(), bins + 1)
        if logy
        else np.linspace(ys.min(), ys.max(), bins + 1)
    )
    counts, _, _ = np.histogram2d(xs, ys, bins=[xe, ye])
    z = counts.T.copy()
    z[z == 0] = np.nan
    xc = np.sqrt(xe[:-1] * xe[1:]) if logx else (xe[:-1] + xe[1:]) / 2
    yc = np.sqrt(ye[:-1] * ye[1:]) if logy else (ye[:-1] + ye[1:]) / 2
    st = _style.get_style()
    shown = np.log10(z) if log_counts else z
    top = float(np.nanmax(shown))
    ticks = np.linspace(float(np.nanmin(shown)), top, 5)
    colorbar = {"title": {"text": "count", "font": {"size": 11, "color": st.ink_2}}, "thickness": 12,
                "outlinewidth": 0, "tickfont": {"size": 10, "color": st.muted}}  # fmt: skip
    if log_counts:
        colorbar.update(
            tickmode="array", tickvals=ticks.tolist(), ticktext=[f"{10**t:.0f}" for t in ticks]
        )
    fig = go.Figure(go.Heatmap(
        x=xc, y=yc, z=shown, colorscale=_style.sequential_scale(), colorbar=colorbar, customdata=z,
        hovertemplate=f"{x}=%{{x:.4g}}<br>{y}=%{{y:.4g}}<br>count=%{{customdata:.0f}}<extra></extra>",
    ))  # fmt: skip
    if logx:
        apply_log(fig, "x", lo, hi)
    if logy:
        apply_log(fig, "y", lo_y, hi_y)
    return finish(fig, title, subtitle or f"n = {xs.size:,} points in {bins} x {bins} cells",
                  xlabel=xlabel or x, ylabel=ylabel or y, height=height, legend=False, width=width)  # fmt: skip


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
    width: int | None = None,
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
        width: Figure width in px; ``None`` fills the container.

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
                  legend=bool(group), width=width)  # fmt: skip


def plot_scatter_matrix(
    frame: pd.DataFrame,
    title: str = "",
    subtitle: str = "",
    *,
    columns: Sequence[str] | None = None,
    bins: int = 30,
    cell: int = 150,
    height: int | None = None,
    width: int | None = None,
) -> go.Figure:
    """Pairwise scatter of up to 6 numeric columns with histograms on the diagonal.

    Args:
        frame: DataFrame with numeric columns.
        title: Figure title.
        subtitle: Muted line under the title.
        columns: Columns to include (default: all numeric ones).
        bins: Bins of the diagonal histograms.
        cell: Size of one cell in px.
        height: Total figure height in px; overrides the height computed from ``cell``.
        width: Figure width in px; ``None`` fills the container.

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
        fig,
        title,
        subtitle,
        height=height or max(HEIGHT, 110 + cell * n),
        legend=False,
        width=width,
    )
