"""Prepared figures in, one dropdown figure out.

Use this when the views differ in annotations, shapes or axes. For subsets of the *same* lines,
``plot_series(views=)`` is simpler.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import plotly.graph_objects as go

from . import style as _style
from .layout import HEIGHT, finish

__all__ = ["plot_views", "annotate_views"]

_CARTESIAN = {
    "scatter", "scattergl", "bar", "histogram", "box", "heatmap", "contour", "histogram2d",
    "histogram2dcontour", "candlestick", "ohlc", "waterfall", "funnel", "violin", "image",
}  # fmt: skip

_AXIS_KEYS = ("type", "range", "autorange", "tickformat", "ticksuffix", "tickprefix", "dtick")


def _traces(fig: go.Figure) -> list[Any]:
    """The figure's traces as a list (plotly's stubs type ``fig.data`` as its keys)."""
    return list(fig.data)


def _check_cartesian(name: str, fig: go.Figure) -> None:
    layout = fig.layout
    if layout.updatemenus or layout.sliders or fig.frames:
        raise ValueError(
            f"view {name!r} already has menus, sliders or animation frames; plot_views "
            "would lose them. Pass figures without interactions."
        )
    bad = sorted({t.type for t in _traces(fig) if t.type not in _CARTESIAN})
    if bad:
        raise ValueError(
            f"view {name!r} has non-Cartesian traces {bad}; plot_views is Cartesian only"
        )


def _axis_state(fig: go.Figure, axis: str) -> dict[str, Any]:
    """Relayout entries that restore one axis to the state this view had."""
    ax = fig.layout[f"{axis}axis"]
    out: dict[str, Any] = {f"{axis}axis.title.text": ax.title.text or ""}
    for key in _AXIS_KEYS:
        value = ax[key]
        if key == "range":
            continue
        out[f"{axis}axis.{key}"] = value
    explicit = ax.range is not None
    out[f"{axis}axis.autorange"] = not explicit
    if explicit:
        out[f"{axis}axis.range"] = list(ax.range)
    if ax.tickvals is not None:
        out[f"{axis}axis.tickmode"] = "array"
        out[f"{axis}axis.tickvals"] = list(ax.tickvals)
        out[f"{axis}axis.ticktext"] = list(ax.ticktext) if ax.ticktext is not None else None
    else:
        out[f"{axis}axis.tickmode"] = "auto"
    return out


def plot_views(
    figures: Mapping[str, go.Figure],
    title: str = "",
    subtitle: str = "",
    *,
    height: int = HEIGHT,
    dropdown_x: float = 0.0,
    dropdown_y: float = 1.0,
    legend: bool | None = None,
) -> go.Figure:
    """Merge prepared figures into one that switches between them with a dropdown.

    Traces, axes (title, range, scale, ticks), annotations, shapes and the legend all switch
    together. The first view opens initially. Source figures are copied, including traces
    that were hidden in them (they stay hidden or ``legendonly`` in their own view).

    Cartesian figures only: existing menus, sliders, animation frames and non-Cartesian
    traces raise rather than silently losing their interactions.

    Args:
        figures: ``{view_name: Figure}``; order is the dropdown order.
        title: Figure title.
        subtitle: Muted line under the title.
        height: Figure height in px.
        dropdown_x: Dropdown x position in paper units.
        dropdown_y: Dropdown y position in paper units (above the plot at 1.0).
        legend: ``None`` shows it when the first view has more than one named trace.

    Returns:
        The merged figure. With a single view no dropdown is added.

    Raises:
        ValueError: no figures, or a figure that is not plain Cartesian.

    Example:
        >>> fig = pl.plot_views({"Raw": raw_figure, "Normalised": normalised_figure})
    """
    if not figures:
        raise ValueError("plot_views needs at least one figure")
    names = list(figures)
    for name in names:
        _check_cartesian(name, figures[name])

    merged = go.Figure()
    owners: list[str] = []
    for name in names:
        for trace in _traces(figures[name]):
            merged.add_trace(go.Figure(trace).data[0])
            owners.append(name)

    def visibility(view: str) -> list[Any]:
        out: list[Any] = []
        offset = 0
        for name in names:
            for trace in _traces(figures[name]):
                original = trace.visible
                out.append((original if original is not None else True) if name == view else False)
                offset += 1
        return out

    buttons = []
    for view in names:
        src = figures[view]
        relayout: dict[str, Any] = {
            "shapes": [s.to_plotly_json() for s in src.layout.shapes],
            "annotations": [a.to_plotly_json() for a in src.layout.annotations],
            **_axis_state(src, "x"),
            **_axis_state(src, "y"),
        }
        buttons.append(
            {"label": view, "method": "update", "args": [{"visible": visibility(view)}, relayout]}
        )

    first = figures[names[0]]
    merged.update_layout(first.layout)
    merged.update_traces(visible=None)  # reset; the first view's state is applied below
    for trace, vis in zip(_traces(merged), buttons[0]["args"][0]["visible"], strict=True):
        trace.visible = vis
    merged.update_layout(shapes=first.layout.shapes, annotations=first.layout.annotations)

    st = _style.get_style()
    if len(names) > 1:
        merged.update_layout(
            updatemenus=[
                {
                    "type": "dropdown",
                    "direction": "down",
                    "x": dropdown_x,
                    "xanchor": "left",
                    "y": dropdown_y,
                    "yanchor": "bottom",
                    "pad": {"t": 0, "b": 6},
                    "bgcolor": st.surface,
                    "bordercolor": st.axis,
                    "font": {"color": st.ink_2, "size": 11},
                    "buttons": buttons,
                }
            ]
        )
    return finish(merged, title, subtitle, height=height, legend=legend, menu=len(names) > 1)


def annotate_views(fig: go.Figure, texts: Mapping[str, str]) -> go.Figure:
    """Add a top-right annotation that changes with the dropdown of an existing figure.

    Modifies and returns ``fig``. Works on figures from :func:`plot_views` and from
    ``plot_series(views=)``; annotations the figure already has are kept in every view.

    Args:
        fig: A figure with a dropdown.
        texts: ``{view_name: text}``; use ``<br>`` for line breaks. Views not listed get no
            extra annotation.

    Raises:
        ValueError: ``fig`` has no dropdown, or ``texts`` names a view it does not have.
    """
    if not fig.layout.updatemenus:
        raise ValueError("fig has no dropdown; build it with plot_views or plot_series(views=)")
    menu = fig.layout.updatemenus[0]
    labels = [b.label for b in menu.buttons]
    unknown = [k for k in texts if k not in labels]
    if unknown:
        raise ValueError(f"unknown views {unknown}; the dropdown has {labels}")

    st = _style.get_style()
    base = [a.to_plotly_json() for a in fig.layout.annotations]
    extra_for = {
        view: {
            "xref": "paper", "x": 1, "xanchor": "right", "yref": "paper", "y": 1,
            "yanchor": "bottom", "showarrow": False, "align": "right", "text": text,
            "font": {"size": 11, "color": st.ink_2, "family": st.mono},
        }
        for view, text in texts.items()
    }  # fmt: skip

    new_buttons = []
    for b in menu.buttons:
        args = list(b.args)
        relayout = dict(args[1]) if len(args) > 1 else {}
        current = list(relayout.get("annotations", base))
        if b.label in extra_for:
            current = [a for a in current if a not in extra_for.values()] + [extra_for[b.label]]
        relayout["annotations"] = current
        args = [args[0], relayout]
        new_buttons.append({"label": b.label, "method": b.method, "args": args})
    menu.buttons = new_buttons
    first = new_buttons[0]["args"][1]["annotations"]
    fig.layout.annotations = first
    return fig
