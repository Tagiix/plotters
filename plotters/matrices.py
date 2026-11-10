"""Matrix charts. Convention D: rows and columns are already the two axes, cells are the values.

These colour by **value** (a rank would discard magnitude: a correlation of 0.97 and one of
-0.02 can both be "rank 2"). For rank colouring see :func:`plotters.plot_rank_heatmap`.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import _color
from . import style as _style
from ._errors import convention_error
from .layout import HEIGHT, apply_log, finish

__all__ = ["plot_matrix", "plot_correlation", "plot_table_heatmap", "plot_contours"]

_MAX_ANNOTATED_CELLS = 900
_DARK_INK = "#0e1116"
_LIGHT_INK = "#f7f8fa"


def _require_frame(frame: object, caller: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise convention_error(caller, "D", frame)
    return frame


def _ramp_color(ramp: tuple[str, ...], t: float) -> str:
    """Colour at position ``t`` in [0, 1] along an evenly spaced ramp (linear in sRGB)."""
    t = min(1.0, max(0.0, t)) * (len(ramp) - 1)
    i = min(int(t), len(ramp) - 2)
    frac = t - i
    a, b = _color.hex_to_rgb(ramp[i]), _color.hex_to_rgb(ramp[i + 1])
    return _color.rgb_to_hex(tuple(x + (y - x) * frac for x, y in zip(a, b, strict=True)))  # type: ignore[arg-type]


def cell_ink(fill: str) -> str:
    """Text colour for a cell, chosen from the fill's **measured luminance**.

    A position threshold fails on the dark-mode diverging ramp, whose midpoint is dark: a rule
    like "light text above the middle" put dark text on a dark fill. Measuring the luminance of
    each actual fill and taking the inkier of two inks avoids that.

    Args:
        fill: Hex colour of the cell.

    Returns:
        A near-black or near-white hex colour, whichever is further in luminance from ``fill``.
    """
    lum = _color.relative_luminance(fill)
    dark_gap = abs(lum - _color.relative_luminance(_DARK_INK))
    light_gap = abs(lum - _color.relative_luminance(_LIGHT_INK))
    return _DARK_INK if dark_gap >= light_gap else _LIGHT_INK


def luminance_gap(fill: str) -> float:
    """Luminance difference between a fill and the ink :func:`cell_ink` picks for it."""
    return abs(_color.relative_luminance(fill) - _color.relative_luminance(cell_ink(fill)))


def _format(value: float, fmt: str) -> str:
    return (
        ""
        if value is None or (isinstance(value, float) and math.isnan(value))
        else format(value, fmt)
    )


def _grid(
    frame: pd.DataFrame,
    values: np.ndarray,
    colors: list[list[str]],
    texts: list[list[str]],
    zmin: float,
    zmax: float,
    scale: list[list[object]],
    title: str,
    subtitle: str,
    annotate: bool,
    hover: str,
    height: int | None,
    colorbar: bool,
    xlabel: str,
    ylabel: str,
) -> go.Figure:
    st = _style.get_style()
    xs = [str(c) for c in frame.columns]
    ys = [str(i) for i in frame.index]
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=xs,
            y=ys,
            zmin=zmin,
            zmax=zmax,
            colorscale=scale,
            xgap=1,
            ygap=1,
            hovertemplate=hover,
            showscale=colorbar,
            colorbar={
                "thickness": 12,
                "outlinewidth": 0,
                "tickfont": {"size": 10, "color": st.muted},
            },
        )  # fmt: skip
    )
    if annotate and values.size <= _MAX_ANNOTATED_CELLS:
        for r, y in enumerate(ys):
            for c, x in enumerate(xs):
                if texts[r][c]:
                    fig.add_annotation(
                        x=x, y=y, text=texts[r][c], showarrow=False,
                        font={"size": 11, "color": cell_ink(colors[r][c]), "family": st.mono},
                    )  # fmt: skip
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(showgrid=False, type="category", side="bottom")
    h = height or int(min(900, max(HEIGHT, 170 + 34 * len(ys))))
    return finish(fig, title, subtitle, xlabel=xlabel, ylabel=ylabel, height=h, legend=False)


def plot_matrix(
    frame: pd.DataFrame,
    title: str = "",
    subtitle: str = "",
    *,
    diverging: bool | None = None,
    center: float = 0.0,
    mask_diagonal: bool = False,
    fmt: str = ".2f",
    annotate: bool = True,
    xlabel: str = "",
    ylabel: str = "",
    height: int | None = None,
) -> go.Figure:
    """Any wide or square frame on its own values, every cell written in.

    The scale is chosen from the data. A frame that straddles ``center`` gets the two diverging
    poles with the range forced **symmetric** about ``center``, otherwise the neutral midpoint
    would lie about where zero is. One that does not straddle gets the single-hue ramp.

    Args:
        frame: Rows and columns are the two axes; cells are the values.
        title: Figure title.
        subtitle: Muted line under the title.
        diverging: ``True`` / ``False`` to force the scale; ``None`` decides from the data.
        center: The neutral value of a diverging scale.
        mask_diagonal: Blank the diagonal (it is trivial for e.g. a correlation matrix).
        fmt: Python format spec for the cell text.
        annotate: Write the value in each cell (skipped above 900 cells).
        xlabel: X axis title.
        ylabel: Y axis title.
        height: Figure height; defaults to a size that grows with the number of rows.

    Raises:
        ConventionError: ``frame`` is not a DataFrame.
        ValueError: ``mask_diagonal`` on a frame that is not square.
    """
    frame = _require_frame(frame, "plot_matrix")
    values = frame.to_numpy(dtype=float).copy()
    if mask_diagonal:
        if values.shape[0] != values.shape[1]:
            raise ValueError("mask_diagonal needs a square frame")
        np.fill_diagonal(values, np.nan)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("the frame has no finite values")
    lo, hi = float(finite.min()), float(finite.max())
    use_div = (lo < center < hi) if diverging is None else diverging
    st = _style.get_style()
    if use_div:
        span = max(abs(lo - center), abs(hi - center)) or 1.0
        zmin, zmax, ramp = center - span, center + span, st.diverging
        scale = _style.diverging_scale()
    else:
        zmin, zmax, ramp = lo, hi if hi > lo else lo + 1.0, st.sequential
        scale = _style.sequential_scale()
    colors = [
        [
            _ramp_color(ramp, (v - zmin) / (zmax - zmin)) if np.isfinite(v) else st.surface
            for v in row
        ]
        for row in values
    ]
    texts = [[_format(float(v), fmt) if np.isfinite(v) else "" for v in row] for row in values]
    return _grid(frame, values, colors, texts, zmin, zmax, scale, title, subtitle, annotate,
                 "%{y} × %{x}: %{z:.4g}<extra></extra>", height, True, xlabel, ylabel)  # fmt: skip


def plot_correlation(
    data: pd.DataFrame,
    title: str = "",
    subtitle: str = "",
    *,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    fixed_range: bool = False,
    fmt: str = ".2f",
    annotate: bool = True,
    height: int | None = None,
) -> go.Figure:
    """Correlation between the columns of a data frame, diverging about 0, diagonal masked.

    Takes the **data**, not a correlation matrix: computing it inside is what keeps the scale
    symmetric and the diagonal masked. The diagonal is 1.0 by construction and would otherwise
    own the whole top of the scale. Pass ``method="spearman"`` on fat-tailed series.

    Args:
        data: Observations in rows, variables in columns (non-numeric columns are ignored).
        title: Figure title.
        subtitle: Muted line under the title.
        method: ``"pearson"``, ``"spearman"`` or ``"kendall"``.
        fixed_range: Use -1 to 1 instead of the data-driven symmetric range, to compare
            several charts at the same scale.
        fmt: Format spec for the cell text.
        annotate: Write the coefficient in each cell.
        height: Figure height in px.

    Raises:
        ConventionError: ``data`` is not a DataFrame, or already looks like a correlation matrix.
        ValueError: fewer than two numeric columns.
    """
    data = _require_frame(data, "plot_correlation")
    numeric = data.select_dtypes("number")
    if numeric.shape[1] < 2:
        raise ValueError("plot_correlation needs at least two numeric columns")
    values = numeric.to_numpy(dtype=float)
    square = values.shape[0] == values.shape[1] and list(numeric.index) == list(numeric.columns)
    if (
        square
        and np.allclose(np.diag(values), 1.0)
        and np.allclose(values, values.T, equal_nan=True)
    ):
        raise convention_error("plot_correlation", "B", data, "That already looks like a correlation "
                               "matrix: pass the data it came from, or use plot_matrix for the matrix.")  # fmt: skip
    corr = numeric.corr(method=method)
    figure = plot_matrix(corr, title, subtitle, diverging=True, center=0.0, mask_diagonal=True, fmt=fmt,
                         annotate=annotate, height=height)  # fmt: skip
    if fixed_range:
        figure.update_traces(zmin=-1, zmax=1)
    return figure


def plot_table_heatmap(
    frame: pd.DataFrame,
    title: str = "",
    subtitle: str = "",
    *,
    normalize: Literal["column", "row"] = "column",
    fmt: str = ".4g",
    height: int | None = None,
) -> go.Figure:
    """A table whose columns are on **different scales**, such as a moments table.

    Each column is shaded on its own min-to-max range so that, say, ``n`` and ``skewness`` can
    share a picture, while the **actual values** are written in the cells. Shading says where a
    value sits within its column; it is not a ranking and does not claim higher is better.
    A constant column carries no information and is drawn in the lightest tone.

    Args:
        frame: Rows and columns are the two axes; cells are the values.
        title: Figure title.
        subtitle: Muted line under the title.
        normalize: ``"column"`` shades each column on its own range, ``"row"`` each row.
        fmt: Format spec for the cell text.
        height: Figure height in px.

    Raises:
        ConventionError: ``frame`` is not a DataFrame.
        ValueError: unknown ``normalize``.
    """
    frame = _require_frame(frame, "plot_table_heatmap")
    if normalize not in ("column", "row"):
        raise ValueError(f"normalize must be 'column' or 'row', got {normalize!r}")
    values = frame.to_numpy(dtype=float)
    axis = 0 if normalize == "column" else 1
    lo = np.nanmin(values, axis=axis, keepdims=True)
    hi = np.nanmax(values, axis=axis, keepdims=True)
    span = np.where(hi > lo, hi - lo, 1.0)
    unit = np.where(hi > lo, (values - lo) / span, 0.0)
    st = _style.get_style()
    colors = [
        [_ramp_color(st.sequential, float(u)) if np.isfinite(u) else st.surface for u in row]
        for row in unit
    ]
    texts = [[_format(float(v), fmt) if np.isfinite(v) else "" for v in row] for row in values]
    figure = _grid(frame, unit, colors, texts, 0.0, 1.0, _style.sequential_scale(), title, subtitle, True,
                   "%{y} × %{x}<extra></extra>", height, False, "", "")  # fmt: skip
    figure.update_traces(
        customdata=values, hovertemplate="%{y} × %{x}: %{customdata:.4g}<extra></extra>"
    )
    return figure


def plot_contours(
    frame: pd.DataFrame,
    title: str = "",
    subtitle: str = "",
    *,
    logx: bool = False,
    logy: bool = False,
    diagonal: bool = False,
    levels: int = 12,
    fmt: str = ".3g",
    xlabel: str = "",
    ylabel: str = "",
    height: int = HEIGHT,
) -> go.Figure:
    """A function of two variables as labelled level curves.

    Takes the grid **already evaluated**, so the caller owns the maths: build ``frame`` with
    ``np.meshgrid`` and whatever function, then pass it. Index is y, columns are x, cells are the
    value. Contours rather than a 3-D surface on purpose: a surface hides its own far side and
    cannot be read off, while labelled level curves give exact values.

    Args:
        frame: Index = y values, columns = x values (numeric), cells = f(x, y).
        title: Figure title.
        subtitle: Muted line under the title.
        logx: Logarithmic x axis (needs positive columns).
        logy: Logarithmic y axis (needs positive index).
        diagonal: Draw ``y = x``, the line where the two arguments agree: the reference for any
            ``f(a, b)`` that should vanish at ``a = b``.
        levels: Approximate number of contour levels.
        fmt: d3 format for the level labels.
        xlabel: X axis title.
        ylabel: Y axis title.
        height: Figure height in px.

    Raises:
        ConventionError: ``frame`` is not a DataFrame.
        ValueError: non-numeric axes, or a log axis with non-positive values.
    """
    frame = _require_frame(frame, "plot_contours")
    try:
        x = frame.columns.to_numpy(dtype=float)
        y = frame.index.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("plot_contours needs numeric index (y) and columns (x)") from exc
    st = _style.get_style()
    fig = go.Figure(
        go.Contour(
            z=frame.to_numpy(dtype=float),
            x=x,
            y=y,
            ncontours=levels,
            showscale=False,
            contours={
                "coloring": "lines",
                "showlabels": True,
                "labelformat": fmt,
                "labelfont": {"size": 10, "color": st.ink_2, "family": st.mono},
            },
            line={"width": 1.8},
            colorscale=_style.sequential_scale(start=0.4),
            hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<br>f=%{z:.4g}<extra></extra>",
        )  # fmt: skip
    )
    if diagonal:
        lo, hi = max(x.min(), y.min()), min(x.max(), y.max())
        if lo < hi:
            fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x", showlegend=False,
                            line={"color": st.negative, "width": 1.4, "dash": "dash"})  # fmt: skip
    for axis, flag, arr in (("x", logx, x), ("y", logy, y)):
        if flag:
            if (arr <= 0).any():
                raise ValueError(
                    f"log{axis} needs strictly positive {'columns' if axis == 'x' else 'index'}"
                )
            apply_log(fig, axis, float(arr.min()), float(arr.max()))  # type: ignore[arg-type]
    return finish(fig, title, subtitle, xlabel=xlabel, ylabel=ylabel, height=height, legend=False)
