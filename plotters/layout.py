"""Per-figure chrome: title, subtitle, legend placement, margins, and log-axis helpers.

:func:`finish` is the only place that sets these, so no plotter can drift from the others.
All pixel maths lives here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import plotly.graph_objects as go

from . import style as _style
from .notebook import _maybe_display

__all__ = ["finish", "apply_log", "log_ticks", "HEIGHT"]

HEIGHT = 420
_TOP_MARGIN_PX = 70
_CHAR_PX = 6.6  # average glyph width at the 11px legend font
_ENTRY_PAD_PX = 34  # marker swatch plus gap per legend entry
_LEGEND_WRAP_PX = 760  # width at which a legend row wraps
_LEGEND_ROW_PX = 22
_MENU_PX = 58  # extra top margin for a dropdown above the plot
_BOTTOM_LEGEND_MAX = (
    12  # past this many entries the legend moves to a scrolling column on the right
)


def _legend_labels(fig: go.Figure) -> list[str]:
    """Names of the traces that will actually draw a legend entry.

    ``showlegend`` is tested with ``is not False`` because it is ``None`` by default
    (meaning shown). A ``visible=False`` trace draws no entry, so it must not be counted:
    a view-switching figure hides every view but the first and would otherwise reserve
    hundreds of pixels of bottom margin.
    """
    return [
        getattr(trace, "name", "") or ""
        for trace in fig.data
        if getattr(trace, "showlegend", None) is not False
        and getattr(trace, "name", None)
        and getattr(trace, "visible", True) is not False
    ]


def _legend_layout(
    labels: Sequence[str], height: int, slider: bool, top: int = _TOP_MARGIN_PX
) -> tuple[int, float]:
    """Bottom margin (px) and legend ``y`` (paper fraction) for a legend under the plot.

    The offset below the axis is fixed in pixels and converted to plotly's
    fraction-of-plot-area, so the legend sits the same distance under the plot at any
    figure height; a fixed fraction drifts on tall figures.
    """
    rows = max(
        1,
        math.ceil(sum(len(n) * _CHAR_PX + _ENTRY_PAD_PX for n in labels) / _LEGEND_WRAP_PX),
    )
    margin_b = (140 if slider else 88) + _LEGEND_ROW_PX * (rows - 1)
    offset_px = 104 if slider else 52
    plot_px = max(height - top - margin_b, 1)
    return margin_b, -(offset_px / plot_px)


def finish(
    fig: go.Figure,
    title: str,
    subtitle: str = "",
    *,
    xlabel: str = "",
    ylabel: str = "",
    height: int = HEIGHT,
    width: int | None = None,
    slider: bool = False,
    legend: bool | None = None,
    legend_position: Literal["auto", "bottom", "right"] = "auto",
    menu: bool = False,
) -> go.Figure:
    """Apply the shared look: left-aligned title, muted subtitle, hairline grid, legend below.

    Call it last in a plotter, after the traces exist. The legend goes *under* the plot
    because the title is left-aligned and plotly's default top-right corner legend lands on
    top of it as soon as there is a subtitle.

    Args:
        fig: Figure to style; modified in place and returned.
        title: Figure title.
        subtitle: Smaller muted line under the title.
        xlabel: X axis title; empty leaves any existing title untouched.
        ylabel: Y axis title; empty leaves any existing title untouched.
        height: Figure height in px.
        width: Figure width in px; ``None`` lets plotly fill the container.
        slider: Reserve room for a range slider under the axis.
        legend: ``None`` shows the legend when more than one trace would appear in it
            (one series needs none, the title names it). ``True`` / ``False`` force it.
        legend_position: ``"bottom"`` puts the legend under the plot, ``"right"`` in a
            vertical column that scrolls when long (the plot keeps its height). ``"auto"``
            uses the right column past 12 entries.
        menu: Reserve room above the plot for a dropdown.

    Returns:
        The same figure, for chaining.

    Example:
        >>> fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4], name="a"))
        >>> fig = pl.finish(fig, "Title", "subtitle", ylabel="y")
    """
    st = _style.get_style()
    muted_sub = f'<sub><span style="color:{st.muted}">{subtitle}</span></sub>'
    heading = muted_sub if subtitle and not title else title
    if subtitle and title:
        heading = f"{title}<br>{muted_sub}"

    labels = _legend_labels(fig)
    show = (len(labels) > 1) if legend is None else legend
    top = _TOP_MARGIN_PX + (_MENU_PX if menu else 0)
    side = legend_position
    if side == "auto":
        side = "right" if len(labels) > _BOTTOM_LEGEND_MAX else "bottom"

    right_margin = 30
    if show and side == "bottom":
        margin_b, legend_y = _legend_layout(labels, height, slider, top)
        legend_box = {
            "orientation": "h",
            "yanchor": "top",
            "y": legend_y,
            "x": 0,
            "xanchor": "left",
        }
    else:
        margin_b = 90 if slider else 50
        widest = max((len(n) for n in labels), default=0)
        right_margin = int(min(320, max(120, widest * _CHAR_PX + 70))) if show else 30
        legend_box = {
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "x": 1.01,
            "xanchor": "left",
        }

    fig.update_layout(
        template=st.name,
        title={
            "text": heading,
            "x": 0,
            "xref": "paper",
            "xanchor": "left",
            "font": {"size": 15, "color": st.ink},
        },
        height=height,
        margin={"l": 60, "r": right_margin, "t": top, "b": margin_b},
        showlegend=show,
        legend={**legend_box, "bgcolor": "rgba(0,0,0,0)"},
    )
    if width is not None:
        fig.update_layout(width=width)
    if xlabel:
        fig.update_xaxes(title_text=xlabel)
    if ylabel:
        fig.update_yaxes(title_text=ylabel)
    _maybe_display(fig)
    return fig


# Anchors per decade, tried from sparse to dense until at least four ticks fall in range.
_LOG_ANCHORS = ((1, 2, 5), (1, 2, 3, 5, 7), (1, 1.5, 2, 3, 4, 5, 6, 8))


def log_ticks(low: float, high: float) -> tuple[list[float], list[str]]:
    """``(tickvals, ticktext)`` for a log axis, anchored at 1, 2 and 5 per decade (denser when the range is narrow).

    Left alone, plotly labels the *minor* ticks of a log axis that crosses 1 and drops the
    leading ``0.``, so 0.2 to 10 reads ``2 3 4 5 6 7 8 9 1 2 3 4``. Writing the full label
    at 1, 2 and 5 of each decade avoids that.

    Args:
        low: Smallest data value shown (must be > 0).
        high: Largest data value shown.

    Returns:
        ``tickvals`` in data units (not yet log10) and their ``ticktext`` labels.
    """
    if low <= 0 or high <= 0:
        raise ValueError("a log axis needs strictly positive bounds")
    lo_dec, hi_dec = math.floor(math.log10(low)), math.ceil(math.log10(high))
    vals: list[float] = []
    for anchors in _LOG_ANCHORS:
        vals = [
            round(a * 10.0**d, 12)
            for d in range(lo_dec, hi_dec + 1)
            for a in anchors
            if low / 1.0001 <= a * 10.0**d <= high * 1.0001
        ]
        if len(vals) >= 4:
            break
    if len(vals) < 2:  # a very narrow range between anchors: fall back to its two ends
        vals = [low, high]
    return vals, [f"{v:g}" for v in vals]


def apply_log(
    fig: go.Figure,
    axis: Literal["x", "y"],
    low: float,
    high: float,
    *,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Make an axis logarithmic with readable ticks and an explicit range.

    Use this instead of ``type="log"`` whenever data crosses 1. Plotly mixes units on a log
    axis: ``range`` is in log10 while ``tickvals`` and the trace data are in data units
    (checked by rendering on plotly 7.1; an earlier draft of the spec assumed log10 ticks).
    This function writes the ticks in data units and converts only the range, which avoids
    autorange stretching decades to reach a tick.

    Args:
        fig: Figure to modify in place.
        axis: ``"x"`` or ``"y"``.
        low: Lower bound in data units (> 0).
        high: Upper bound in data units.
        row: Subplot row, for faceted figures.
        col: Subplot column, for faceted figures.

    Returns:
        The same figure.
    """
    vals, text = log_ticks(low, high)
    update = fig.update_xaxes if axis == "x" else fig.update_yaxes
    update(
        type="log",
        tickmode="array",
        tickvals=vals,
        ticktext=text,
        range=[math.log10(low), math.log10(high)],
        row=row,
        col=col,
    )
    return fig
