"""Prepared figures in, one dropdown figure out.

Use this when the views differ in annotations, shapes or axes. For subsets of the *same* lines,
``plot_series(views=)`` is simpler.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import style as _style
from .layout import HEIGHT, finish

__all__ = ["plot_views", "annotate_views", "plot_grid"]

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


def _axis_props(fig: go.Figure, axis: str) -> dict[str, Any]:
    """The axis settings worth carrying into a grid panel (keyword form for ``update_xaxes``)."""
    ax = fig.layout[f"{axis}axis"]
    props: dict[str, Any] = {"title_text": ax.title.text or ""}
    for key in ("type", "tickformat", "ticksuffix", "tickprefix", "dtick", "categoryorder"):
        if ax[key] is not None:
            props[key] = ax[key]
    if ax.range is not None:
        props["range"] = list(ax.range)
    if ax.autorange == "reversed":
        props["autorange"] = "reversed"
    if ax.tickvals is not None:
        props.update(tickmode="array", tickvals=list(ax.tickvals))
        if ax.ticktext is not None:
            props["ticktext"] = list(ax.ticktext)
    if ax.categoryarray is not None:
        props["categoryarray"] = list(ax.categoryarray)
    return props


def _remap(ref: str | None, axis_id: str, letter: str) -> str | None:
    """Point a shape / annotation reference ('x', 'x domain', 'paper') at one grid panel's axis."""
    if ref is None:
        return None
    if ref == "paper":
        return f"{axis_id} domain"
    if ref.startswith(letter):
        return axis_id + ref[len(letter) :]
    return ref


def plot_grid(
    figures: Mapping[str, go.Figure] | Sequence[go.Figure],
    title: str = "",
    subtitle: str = "",
    *,
    ncols: int = 2,
    panel_height: int = 320,
    shared_x: bool = False,
    shared_y: bool = False,
    legend: bool | None = None,
) -> go.Figure:
    """Arrange finished figures side by side in one grid, each keeping its own axes.

    The static counterpart of :func:`plot_views` (which swaps figures with a dropdown). Traces, axis
    titles and scales, shapes and annotations are carried into each panel. A legend entry is shown
    once per name, so the same series in several panels is not listed repeatedly.

    Cartesian figures only: existing menus, sliders, animation frames and non-Cartesian traces
    raise rather than being half-copied.

    Args:
        figures: ``{panel title: Figure}``, or a list of figures (each figure's own title becomes
            its panel title).
        title: Title of the whole grid.
        subtitle: Muted line under the title.
        ncols: Panels per row.
        panel_height: Height of one panel row in px.
        shared_x: Share the x axis across panels.
        shared_y: Share the y axis across panels.
        legend: ``None`` shows the legend when any named trace asks for one.

    Returns:
        One figure containing every panel.

    Raises:
        ValueError: no figures, or a figure that is not plain Cartesian.

    Example:
        >>> pl.plot_grid({"Raw": raw, "Normalised": normalised}, ncols=2)
    """
    items = (
        list(figures.items())
        if isinstance(figures, Mapping)
        else [
            (
                re.sub(r"<[^>]+>", "", (f.layout.title.text or "").split("<br>")[0])
                or f"Panel {i + 1}",
                f,
            )
            for i, f in enumerate(figures)
        ]
    )
    if not items:
        raise ValueError("plot_grid needs at least one figure")
    for name, fig in items:
        _check_cartesian(name, fig)
    rows = -(-len(items) // ncols)
    st = _style.get_style()
    grid = make_subplots(rows=rows, cols=min(ncols, len(items)), subplot_titles=[n for n, _ in items],
                         horizontal_spacing=0.07, vertical_spacing=min(0.16, 0.7 / rows),
                         shared_xaxes=shared_x, shared_yaxes=shared_y)  # fmt: skip
    grid.update_annotations(font={"size": 11, "color": st.ink_2})  # the panel titles
    seen: set[str] = set()
    for k, (_, fig) in enumerate(items):
        r, c = k // ncols + 1, k % ncols + 1
        xid, yid = ("x" if k == 0 else f"x{k + 1}"), ("y" if k == 0 else f"y{k + 1}")
        for trace in _traces(fig):
            copy = go.Figure(trace).data[0]
            name = getattr(copy, "name", None)
            if name:
                if name in seen:
                    copy.showlegend = False
                seen.add(name)
            grid.add_trace(copy, row=r, col=c)
        grid.update_xaxes(row=r, col=c, **_axis_props(fig, "x"))
        grid.update_yaxes(row=r, col=c, **_axis_props(fig, "y"))
        for shape in fig.layout.shapes:
            data = shape.to_plotly_json()
            data["xref"], data["yref"] = (
                _remap(data.get("xref"), xid, "x"),
                _remap(data.get("yref"), yid, "y"),
            )
            grid.add_shape(data)
        for note in fig.layout.annotations:
            data = note.to_plotly_json()
            data["xref"], data["yref"] = (
                _remap(data.get("xref"), xid, "x"),
                _remap(data.get("yref"), yid, "y"),
            )
            grid.add_annotation(data)
    show = (
        legend
        if legend is not None
        else any(
            getattr(t, "showlegend", None) is not False and getattr(t, "name", None)
            for t in _traces(grid)
        )
    )
    return finish(grid, title, subtitle, height=max(HEIGHT, 140 + panel_height * rows), legend=show)
