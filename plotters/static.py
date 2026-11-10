"""Print figures with matplotlib, in the same palette as the plotly charts.

Not imported by ``plotters/__init__.py``: it needs matplotlib, which the plotly side does not.
Use ``from plotters import static as ps``::

    import plotters as pl
    from plotters import static as ps

    pl.set_style(dark=False)  # same palette as the plotly charts
    ps.use_print_style()  # rcParams: serif, 9pt, the palette's inks
    figure = ps.plot_lines(frame, title="...", ylabel="...", label_last=True)
    ps.save(figure, "figures/thing.png")  # 300 dpi, closes the figure

What changes for print, and why:

* ``vary_dash`` defaults ON. On screen the legend isolates a series on click; a printed page has no
  legend to click, and photocopied or read by a colourblind reader four solid lines are one line.
* End-of-line labels are nudged apart. Two curves finishing at the same height overprint and both
  become unreadable, which is worse than the legend they replace.
* Subtitles wrap to the axes width. A figure is embedded at its own size, so an overlong subtitle
  runs off the page; nothing clips it.
* Cell ink comes from measured luminance, as in :mod:`plotters.matrices`.
* ``percent=True`` writes a bare ``%``. The PNG is a raster, so matplotlib draws that text and a
  LaTeX escape would print the backslash.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from typing import Any

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from . import style as _style
from ._types import SeriesLike
from .frames import _one_column, _require_columns, as_series_dict, pick
from .matrices import _ramp_color, cell_ink

__all__ = [
    "use_print_style",
    "save",
    "figure_size",
    "plot_lines",
    "plot_points",
    "plot_ranked_bars",
    "plot_heatmap",
    "plot_function",
    "plot_stems",
    "plot_panels",
]

TEXTWIDTH_IN = (21.0 - 2 * 2.2) / 2.54  # a4paper with 2.2 cm margins
_DASHES = ((None, None), (6, 2), (1.2, 2), (6, 2, 1.2, 2), (9, 3), (9, 3, 1.2, 3))
_MARKERS = ("o", "s", "D", "^", "P", "v", "*", "X")


def figure_size(width: float = 0.86, aspect: float = 0.62) -> tuple[float, float]:
    """Figure size in inches for a fraction of ``\\textwidth`` (a4paper, 2.2 cm margins).

    **Size here, never with** ``\\includegraphics[scale=]``: scaling in LaTeX scales the figure's text
    too, so two figures at different scales end up with different apparent font sizes on the
    same page.

    Args:
        width: Fraction of the text width (0.86 is a comfortable single column).
        aspect: Height divided by width.

    Returns:
        ``(width_in, height_in)`` for ``figsize=``.
    """
    return TEXTWIDTH_IN * width, TEXTWIDTH_IN * width * aspect


def use_print_style() -> None:
    """Set matplotlib's rcParams for print: serif, 9 pt, hairline grid, the palette's inks.

    Call after :func:`plotters.set_style` so the current palette is used.
    """
    st = _style.get_style()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": st.surface,
            "axes.facecolor": st.surface,
            "savefig.facecolor": st.surface,
            "axes.edgecolor": st.axis,
            "axes.labelcolor": st.ink_2,
            "text.color": st.ink,
            "xtick.color": st.muted,
            "ytick.color": st.muted,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": st.grid,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=list(_style.palette())),
            "legend.frameon": False,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "lines.linewidth": 1.6,
            "figure.constrained_layout.use": True,
        }
    )


def save(figure: Figure, path: str | PathLike[str], *, dpi: int = 300) -> None:
    """Write the figure at ``dpi`` and close it, so figures do not pile up in memory.

    Args:
        figure: A matplotlib figure from one of the ``plot_*`` functions.
        path: Output file (the extension picks the format).
        dpi: Resolution for raster formats.
    """
    figure.savefig(str(path), dpi=dpi)
    plt.close(figure)


def _wrap(figure: Figure, text: str, width_px: float, size: float) -> str:
    """Wrap ``text`` so no line is wider than ``width_px`` when drawn at ``size`` points."""
    renderer = figure.canvas.get_renderer()  # type: ignore[attr-defined]
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        probe = figure.text(0, 0, trial, fontsize=size)
        too_wide = probe.get_window_extent(renderer).width > width_px
        probe.remove()
        if too_wide and current:
            lines.append(current)
            current = word
        else:
            current = trial
    return "\n".join([*lines, current] if current else lines)


def _decorate(
    figure: Figure,
    ax: Any,
    title: str,
    subtitle: str,
    xlabel: str,
    ylabel: str,
    percent: bool = False,
) -> None:
    """Left-aligned title, muted subtitle wrapped to the axes width, axis labels, % formatting."""
    st = _style.get_style()
    figure.canvas.draw()
    width = ax.get_window_extent().width
    pad = 4.0
    if subtitle:
        wrapped = _wrap(figure, subtitle, width, 8)
        sub = ax.text(0, 1.0, wrapped, transform=ax.transAxes, va="bottom", ha="left", fontsize=8,
                      color=st.muted, linespacing=1.25)  # fmt: skip
        height_pt = (wrapped.count("\n") + 1) * 8 * 1.25
        pad += height_pt + 3
        sub.set_position((0, 1.0))
        sub.set_transform(
            ax.transAxes + mtransforms.ScaledTranslation(0, 3 / 72, figure.dpi_scale_trans)
        )
    if title:
        ax.set_title(title, loc="left", pad=pad, fontsize=10, color=st.ink)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if percent:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v * 100:g}%")
        )  # bare %, never \\%


def _label_ends(ax: Any, ends: list[tuple[float, float, str, str]], min_gap: float = 0.05) -> None:
    """Write each series' name just past its last point, nudged apart so none overprint.

    Labels are sorted by height and pushed apart to at least ``min_gap`` of the axes height; two
    curves finishing at the same y would otherwise draw their names on top of each other.
    """
    inv = ax.transAxes.inverted()
    ordered = sorted(ends, key=lambda e: e[1])
    ys = [float(inv.transform(ax.transData.transform((0, e[1])))[1]) for e in ordered]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    overflow = ys[-1] - 1.0
    if overflow > 0:  # push the stack back down so the top label stays inside the axes
        ys = [y - overflow for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - min_gap)
    for (x, _y, name, color), y_axes in zip(ordered, ys, strict=True):
        x_axes = float(inv.transform(ax.transData.transform((x, 0)))[0])
        ax.annotate(name, xy=(x_axes, y_axes), xycoords="axes fraction", xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8, color=color, annotation_clip=False)  # fmt: skip


def plot_lines(
    data: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    series: Sequence[str] | None = None,
    label_last: bool = False,
    vary_dash: bool = True,
    percent: bool = False,
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """One line per series (convention A).

    Args:
        data: Series, wide DataFrame or ``{label: Series}``.
        title: Left-aligned title.
        subtitle: Muted line under the title; wrapped to the axes width.
        series: Subset and order of series (assigns colour slots).
        label_last: Write each name at the end of its line, nudged apart, instead of a legend.
        vary_dash: Vary the dash pattern per series (default ON for print: there is no legend to
            click, and colour alone fails for a colourblind reader or a photocopy).
        percent: Format the y axis as a percentage; writes a bare ``%``.
        xlabel: X axis title.
        ylabel: Y axis title.
        figsize: Size in inches; default :func:`figure_size`.
    """
    frames = pick(as_series_dict(data, _caller="plot_lines"), series)
    figure, ax = plt.subplots(figsize=figsize or figure_size())
    ends = []
    for i, (name, s) in enumerate(frames.items()):
        dash, _ = _style.line_style(i, vary_dash=vary_dash)
        pattern = (
            _DASHES[_style._DASHES.index(dash) % len(_DASHES)] if dash != "solid" else (None, None)
        )
        line, *_ = ax.plot(s.index, s.to_numpy(), color=_style.color(i), label=str(name))
        if pattern[0] is not None:
            line.set_dashes(pattern)
        valid = s.dropna()
        if len(valid):
            ends.append((valid.index[-1], float(valid.iloc[-1]), str(name), line.get_color()))
    if any(isinstance(s.index, pd.DatetimeIndex) for s in frames.values()):
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    if label_last and len(frames) > 1:
        left, right = ax.get_xlim()
        ax.set_xlim(left, right + 0.14 * (right - left))  # room for the names past the last point
        figure.canvas.draw()
        _label_ends(
            ax,
            [
                (mdates.date2num(x) if isinstance(x, pd.Timestamp) else x, y, n, c)
                for x, y, n, c in ends
            ],
        )
    elif len(frames) > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=min(len(frames), 4))
    _decorate(figure, ax, title, subtitle, xlabel, ylabel, percent)
    return figure


def _pareto(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    keep = np.zeros(len(x), dtype=bool)
    best = np.inf
    for i in np.lexsort((y, x)):
        if y[i] < best:
            keep[i] = True
            best = y[i]
    return keep


def plot_points(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    subtitle: str = "",
    *,
    label: str | None = None,
    group: str | None = None,
    frontier: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Scatter (convention B), optionally with the Pareto front (lower x and lower y win).

    Args:
        frame: Long DataFrame.
        x: Column for x.
        y: Column for y.
        title: Left-aligned title.
        subtitle: Muted wrapped subtitle.
        label: Column written next to each point.
        group: Column splitting points into coloured series.
        frontier: Draw the Pareto front (lower is better on both axes).
        xlabel: X axis title; defaults to the column name.
        ylabel: Y axis title; defaults to the column name.
        figsize: Size in inches.
    """
    _require_columns(frame, "plot_points", x=x, y=y, label=label, group=group)
    st = _style.get_style()
    figure, ax = plt.subplots(figsize=figsize or figure_size())
    xs, ys = frame[x].to_numpy(dtype=float), frame[y].to_numpy(dtype=float)
    gcol = _one_column(frame, group).astype(str).to_numpy() if group else np.full(len(frame), "")
    for i, g in enumerate(pd.unique(gcol)):
        m = gcol == g
        ax.scatter(xs[m], ys[m], s=28, marker=_MARKERS[i % len(_MARKERS)], color=_style.color(i),
                   label=g or None, zorder=3)  # fmt: skip
    if label:
        for xv, yv, t in zip(xs, ys, _one_column(frame, label).astype(str), strict=True):
            ax.annotate(
                t, (xv, yv), xytext=(3, 3), textcoords="offset points", fontsize=7, color=st.muted
            )
    if frontier:
        keep = _pareto(xs, ys)
        order = np.argsort(xs[keep])
        ax.plot(
            xs[keep][order],
            ys[keep][order],
            ls="--",
            lw=1.0,
            color=st.muted,
            label="Pareto front",
            zorder=2,
        )
    if group or frontier:
        ax.legend(loc="best")
    _decorate(figure, ax, title, subtitle, xlabel or x, ylabel or y)
    return figure


def plot_ranked_bars(
    values: pd.Series | Mapping[str, float],
    title: str = "",
    subtitle: str = "",
    *,
    ascending: bool = False,
    fmt: str = ",.4g",
    xlabel: str = "",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Sorted horizontal bars with the value written at each tip: the ranking figure for print.

    Args:
        values: A Series indexed by label, or a ``{label: number}`` dict.
        title: Left-aligned title.
        subtitle: Muted wrapped subtitle.
        ascending: Smallest first (at the top) instead of largest first.
        fmt: Format spec for the written values.
        xlabel: X axis title.
        figsize: Size in inches; default grows with the number of bars.
    """
    s = pd.Series(values, dtype=float).sort_values(ascending=ascending)
    st = _style.get_style()
    h = figsize or (figure_size()[0], max(1.6, 0.9 + 0.24 * len(s)))
    figure, ax = plt.subplots(figsize=h)
    bars = ax.barh([str(i) for i in s.index], s.to_numpy(), color=st.accent, height=0.7, zorder=3)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for bar, v in zip(bars, s, strict=True):
        ax.annotate(format(v, fmt), (bar.get_width(), bar.get_y() + bar.get_height() / 2), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=st.ink_2)  # fmt: skip
    ax.margins(x=0.12)
    _decorate(figure, ax, title, subtitle, xlabel, "")
    return figure


def plot_heatmap(
    frame: pd.DataFrame,
    title: str = "",
    subtitle: str = "",
    *,
    diverging: bool | None = None,
    center: float = 0.0,
    fmt: str = ".2f",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """A matrix on its own values (convention D), every cell written in.

    Same scale rule as :func:`plotters.plot_matrix`: a frame straddling ``center`` gets the diverging
    poles with a symmetric range; otherwise the single-hue ramp. Ink comes from measured luminance.

    Args:
        frame: Wide or square DataFrame.
        title: Left-aligned title.
        subtitle: Muted wrapped subtitle.
        diverging: Force the scale; ``None`` decides from the data.
        center: Neutral value of a diverging scale.
        fmt: Format spec for the cell text.
        figsize: Size in inches.
    """
    st = _style.get_style()
    values = frame.to_numpy(dtype=float)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    use_div = (lo < center < hi) if diverging is None else diverging
    if use_div:
        span = max(abs(lo - center), abs(hi - center)) or 1.0
        vmin, vmax, ramp = center - span, center + span, st.diverging
    else:
        vmin, vmax, ramp = lo, hi if hi > lo else lo + 1.0, st.sequential
    cmap = LinearSegmentedColormap.from_list("plotters", list(ramp))
    figure, ax = plt.subplots(
        figsize=figsize or (figure_size()[0], max(1.8, 0.9 + 0.3 * frame.shape[0]))
    )
    ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(frame.shape[1]), [str(c) for c in frame.columns])
    ax.set_yticks(range(frame.shape[0]), [str(i) for i in frame.index])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for r in range(frame.shape[0]):
        for c in range(frame.shape[1]):
            v = values[r, c]
            if np.isfinite(v):
                fill = _ramp_color(ramp, (v - vmin) / (vmax - vmin))
                ax.text(
                    c,
                    r,
                    format(v, fmt),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=cell_ink(fill),
                )
    _decorate(figure, ax, title, subtitle, "", "")
    return figure


def plot_function(
    functions: Mapping[str, Callable[[np.ndarray], np.ndarray]],
    x: Sequence[float] | np.ndarray,
    title: str = "",
    subtitle: str = "",
    *,
    marks: Sequence[float] = (),
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """The shape of a function (a loss, say), not data.

    Args:
        functions: ``{label: callable}``; each callable takes the x array and returns y.
        x: Where to evaluate the functions.
        title: Left-aligned title.
        subtitle: Muted wrapped subtitle.
        marks: x positions for dotted vertical reference lines.
        xlabel: X axis title.
        ylabel: Y axis title.
        figsize: Size in inches.
    """
    st = _style.get_style()
    xs = np.asarray(x, dtype=float)
    figure, ax = plt.subplots(figsize=figsize or figure_size())
    for i, (name, fn) in enumerate(functions.items()):
        line, *_ = ax.plot(xs, fn(xs), color=_style.color(i), label=name)
        pattern = _DASHES[i % len(_DASHES)]
        if pattern[0] is not None:
            line.set_dashes(pattern)
    for m in marks:
        ax.axvline(m, color=st.muted, lw=0.8, ls=":")
    if len(functions) > 1:
        ax.legend(loc="best")
    _decorate(figure, ax, title, subtitle, xlabel, ylabel)
    return figure


def plot_stems(
    values: Sequence[float] | np.ndarray | pd.Series,
    title: str = "",
    subtitle: str = "",
    *,
    band: Sequence[float] | np.ndarray | None = None,
    xlabel: str = "lag",
    ylabel: str = "acf",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """A correlogram: a stem from zero to each value with a marker at the tip.

    Args:
        values: Coefficients at lags 1 to n.
        title: Left-aligned title.
        subtitle: Muted wrapped subtitle.
        band: Half-width of the significance band **per lag**, so
            :func:`plotters.bartlett_band`'s widening draws itself.
        xlabel: X axis title.
        ylabel: Y axis title.
        figsize: Size in inches.
    """
    st = _style.get_style()
    v = np.asarray(values, dtype=float)
    lag = np.arange(1, len(v) + 1)
    figure, ax = plt.subplots(figsize=figsize or figure_size())
    if band is not None:
        b = np.asarray(band, dtype=float)
        ax.fill_between(lag, -b, b, color=st.accent, alpha=0.15, lw=0, zorder=1)
    ax.axhline(0, color=st.axis, lw=0.8)
    ax.vlines(lag, 0, v, color=st.accent, lw=1.2, zorder=3)
    ax.plot(lag, v, "o", ms=3.5, color=st.accent, zorder=4)
    _decorate(figure, ax, title, subtitle, xlabel, ylabel)
    return figure


def plot_panels(
    panels: Mapping[str, SeriesLike],
    title: str = "",
    subtitle: str = "",
    *,
    ncols: int = 2,
    vary_dash: bool = True,
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """A grid of line panels (``{panel: data}``) with one shared legend and labels on the edges only.

    Args:
        panels: ``{panel name: Series / wide DataFrame / dict}``.
        title: Left-aligned figure title.
        subtitle: Muted wrapped subtitle.
        ncols: Panels per row.
        vary_dash: Vary the dash per series, as in :func:`plot_lines`.
        xlabel: X axis title (bottom row only).
        ylabel: Y axis title (left column only).
        figsize: Size in inches; default grows with the number of rows.
    """
    st = _style.get_style()
    n = len(panels)
    rows = -(-n // ncols)
    w, h = figure_size()
    figure, axes = plt.subplots(rows, ncols, figsize=figsize or (w, max(h, 1.8 * rows + 0.8)), sharex=True,
                                sharey=True, squeeze=False)  # fmt: skip
    normalized = {
        name: as_series_dict(data, _caller="plot_panels") for name, data in panels.items()
    }
    handles: dict[str, Any] = {}
    for k, (name, group) in enumerate(normalized.items()):
        ax = axes[k // ncols][k % ncols]
        for i, (label, s) in enumerate(group.items()):
            dash, _ = _style.line_style(i, vary_dash=vary_dash)
            line, *_ = ax.plot(
                s.index, s.to_numpy(), color=_style.color(i), label=str(label), lw=1.3
            )
            pattern = (
                _DASHES[_style._DASHES.index(dash) % len(_DASHES)]
                if dash != "solid"
                else (None, None)
            )
            if pattern[0] is not None:
                line.set_dashes(pattern)
            handles.setdefault(str(label), line)
        ax.set_title(str(name), loc="left", fontsize=8.5, color=st.ink_2)
    if any(
        isinstance(series.index, pd.DatetimeIndex)
        for group in normalized.values()
        for series in group.values()
    ):
        locator = mdates.AutoDateLocator()
        axes[0][0].xaxis.set_major_locator(locator)
        axes[0][0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for k in range(n, rows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    for c in range(ncols):  # label the lowest visible panel of each column
        lowest = [axes[r][c] for r in range(rows) if axes[r][c].get_visible()]
        if lowest:
            lowest[-1].tick_params(labelbottom=True)
            if xlabel:
                lowest[-1].set_xlabel(xlabel)
    for row in axes:
        if ylabel:
            row[0].set_ylabel(ylabel)
    if len(handles) > 1:
        figure.legend(
            handles.values(), handles.keys(), loc="outside lower center", ncol=min(len(handles), 4)
        )
    if title:
        figure.suptitle(title, x=0.01, ha="left", fontsize=10, color=st.ink)
    if subtitle:
        figure.canvas.draw()
        width = figure.get_size_inches()[0] * figure.dpi * 0.96
        wrapped = _wrap(figure, subtitle, width, 8)
        figure.text(0.01, 0.93, wrapped, ha="left", va="top", fontsize=8, color=st.muted)
    return figure
