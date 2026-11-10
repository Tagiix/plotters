"""Comparison charts. Convention B: a long frame plus column-name strings, one row per observation.

Every ``x``, ``y``, ``label``, ``group``, ``panel``, ``index``, ``columns`` and ``values`` argument is
a **string naming a column**, never data. The exception is :func:`plot_paired_difference`, which is
convention A.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import _color
from . import style as _style
from ._types import MAX_ALL_PAIRS, MAX_SERIES, SeriesLike
from .frames import _one_column, _require_columns, as_series_dict, group_by_prefix, pick
from .layout import HEIGHT, apply_log, finish

__all__ = [
    "plot_tradeoff",
    "plot_rank_heatmap",
    "plot_box_series",
    "plot_small_multiples",
    "plot_paired_difference",
    "plot_scatter",
    "plot_shaded_scatter",
    "plot_interval",
]

_MARKER_SIZE = 9


def _numeric(frame: pd.DataFrame, column: str, caller: str) -> np.ndarray:
    try:
        return _one_column(frame, column).to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{caller}: column {column!r} must be numeric") from exc


def _cap_groups(values: Sequence[object], caller: str, column: str) -> None:
    if len(values) > MAX_ALL_PAIRS:
        raise ValueError(
            f"{caller}: group={column!r} has {len(values)} values; scatter and facets are read pair "
            f"by pair and are capped at {MAX_ALL_PAIRS} (no ordering of 8 hues clears the all-pairs "
            "floors past 3). Use label= to annotate points, groups={family: prefixes} to collapse "
            "values into families, or plot_small_multiples."
        )


def _log_bounds(values: np.ndarray, caller: str, axis: str) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or (finite <= 0).any():
        raise ValueError(f"{caller}: log{axis} needs strictly positive values")
    return float(finite.min()) / 1.25, float(finite.max()) * 1.25


def _bool_pair(flag: bool | tuple[bool, bool], caller: str) -> tuple[bool, bool]:
    if isinstance(flag, bool):
        return flag, flag
    if len(flag) == 2 and all(isinstance(f, bool) for f in flag):
        return flag[0], flag[1]
    raise ValueError(
        f"{caller}: lower_is_better must be a bool or a (bool, bool) pair, got {flag!r}"
    )


def _pareto_mask(x: np.ndarray, y: np.ndarray, lower_x: bool, lower_y: bool) -> np.ndarray:
    """True for the points no other point beats on both objectives."""
    a = x if lower_x else -x
    b = y if lower_y else -y
    order = np.lexsort((b, a))
    keep = np.zeros(len(x), dtype=bool)
    best = math.inf
    for i in order:
        if b[i] < best:
            keep[i] = True
            best = b[i]
    return keep


def _sub(subtitle: str, default: str) -> str:
    return subtitle or default


def plot_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    subtitle: str = "",
    *,
    group: str | None = None,
    label: str | None = None,
    diagonal: bool = False,
    logx: bool = False,
    logy: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    height: int = HEIGHT,
) -> go.Figure:
    """Two columns against each other, optionally split by group or annotated by label.

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
        label: Column written next to each point.
        diagonal: Draw ``y = x``.
        logx: Logarithmic x axis (needs positive values).
        logy: Logarithmic y axis.
        xlabel: X axis title; defaults to the column name.
        ylabel: Y axis title; defaults to the column name.
        height: Figure height in px.

    Raises:
        ConventionError: ``frame`` is not a DataFrame (a Series or dict was passed).
        KeyError: a named column is missing.
        ValueError: more than 3 groups, a non-numeric axis column, or a log axis with
            non-positive values.
    """
    _require_columns(frame, "plot_scatter", x=x, y=y, group=group, label=label)
    xs, ys = _numeric(frame, x, "plot_scatter"), _numeric(frame, y, "plot_scatter")
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
        fig.add_scatter(
            x=xs[mask], y=ys[mask], mode="markers+text" if label else "markers", name=name,
            marker={**_style.marker(i, size=_MARKER_SIZE), "color": _style.color(i) if group else st.accent},
            text=text, textposition="top right", textfont={"size": 10, "color": st.muted},
            showlegend=group is not None,
        )  # fmt: skip
    if diagonal:
        lo, hi = float(min(np.nanmin(xs), np.nanmin(ys))), float(max(np.nanmax(xs), np.nanmax(ys)))
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x", showlegend=False,
                        line={"color": st.muted, "width": 1, "dash": "dot"}, hoverinfo="skip")  # fmt: skip
    if logx:
        apply_log(fig, "x", *_log_bounds(xs, "plot_scatter", "x"))
    if logy:
        apply_log(fig, "y", *_log_bounds(ys, "plot_scatter", "y"))
    return finish(fig, title, subtitle, xlabel=xlabel or x, ylabel=ylabel or y, height=height,
                  legend=group is not None)  # fmt: skip


def plot_shaded_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    shade: str,
    title: str = "",
    subtitle: str = "",
    *,
    label: str | None = None,
    log_shade: bool = False,
    reference: float | None = None,
    logx: bool = False,
    logy: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    height: int = HEIGHT,
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
        label: Column written next to each point.
        log_shade: Colour by ``log10(shade)`` for a geometric ratio (needs positive values);
            the colourbar is still labelled in data units.
        reference: A shade value that is "neutral" (such as 1.0 for a ratio). Switches to the
            diverging ramp centred on it, symmetric in log space when ``log_shade``.
        logx: Logarithmic x axis.
        logy: Logarithmic y axis.
        xlabel: X axis title; defaults to the column name.
        ylabel: Y axis title; defaults to the column name.
        height: Figure height in px.

    Raises:
        ValueError: ``shade`` is not numeric, or ``log_shade`` / ``reference`` with non-positive
            values.
    """
    caller = "plot_shaded_scatter"
    _require_columns(frame, caller, x=x, y=y, shade=shade, label=label)
    xs, ys, sh = (
        _numeric(frame, x, caller),
        _numeric(frame, y, caller),
        _numeric(frame, shade, caller),
    )
    st = _style.get_style()
    values = sh
    if log_shade:
        if (sh[np.isfinite(sh)] <= 0).any():
            raise ValueError(f"{caller}: log_shade needs strictly positive shade values")
        values = np.log10(sh)
    cmin, cmax = float(np.nanmin(values)), float(np.nanmax(values))
    marker: dict[str, object] = {
        "size": _MARKER_SIZE + 1,
        "color": values,
        "line": {"width": 0.6, "color": st.surface},
    }
    if reference is not None:
        center = math.log10(reference) if log_shade else reference
        span = max(abs(cmin - center), abs(cmax - center)) or 1.0
        marker.update(colorscale=_style.diverging_scale(), cmin=center - span, cmax=center + span)
    else:
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
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers+text" if label else "markers", marker=marker, text=text,
        textposition="top right", textfont={"size": 10, "color": st.muted}, customdata=sh,
        hovertemplate=f"{x}=%{{x:.4g}}<br>{y}=%{{y:.4g}}<br>{shade}=%{{customdata:.4g}}<extra></extra>",
    ))  # fmt: skip
    if logx:
        apply_log(fig, "x", *_log_bounds(xs, caller, "x"))
    if logy:
        apply_log(fig, "y", *_log_bounds(ys, caller, "y"))
    return finish(
        fig, title, subtitle, xlabel=xlabel or x, ylabel=ylabel or y, height=height, legend=False
    )


def plot_tradeoff(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    subtitle: str = "",
    *,
    label: str | None = None,
    group: str | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    lower_is_better: bool | tuple[bool, bool] = (True, True),
    connect: str | None = None,
    logx: bool = False,
    logy: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    height: int = HEIGHT,
) -> go.Figure:
    """Two competing objectives and the Pareto frontier: "you cannot have both".

    Points no other point beats on both axes form the frontier, drawn as a dashed line with ringed
    points. With ``group`` (members) and ``groups`` (families), **colour means family and shape
    means member**.

    Args:
        frame: Long DataFrame, one row per candidate.
        x: Column for the first objective.
        y: Column for the second objective.
        title: Figure title.
        subtitle: Muted line under the title; a default states which corner wins.
        label: Column written next to each point (e.g. the model name).
        group: Column naming the member of a family (at most 3 values unless ``groups`` collapses
            them).
        groups: ``{family: prefixes}``. Collapses ``group``'s values into families by the
            **longest** matching prefix, so colour means family and shape means member.
        lower_is_better: Which corner wins: one bool for both axes or ``(x_flag, y_flag)``.
        connect: Numeric column; joins each group's points in order along it, turning a cloud into
            one path per group.
        logx: Logarithmic x axis.
        logy: Logarithmic y axis.
        xlabel: X axis title; defaults to the column name.
        ylabel: Y axis title; defaults to the column name.
        height: Figure height in px.

    Raises:
        ValueError: too many families, a non-numeric axis, or a malformed ``lower_is_better``.
    """
    caller = "plot_tradeoff"
    _require_columns(frame, caller, x=x, y=y, label=label, group=group, connect=connect)
    lx, ly = _bool_pair(lower_is_better, caller)
    xs, ys = _numeric(frame, x, caller), _numeric(frame, y, caller)
    front = _pareto_mask(xs, ys, lx, ly)
    st = _style.get_style()

    members = _one_column(frame, group).astype(str) if group else pd.Series("", index=frame.index)
    if groups is not None:
        if group is None:
            raise ValueError(
                f"{caller}: groups= needs group= to say which column holds the members"
            )
        families = group_by_prefix(members, groups)
    else:
        families = members
    family_names = list(pd.unique(families))
    if group:
        _cap_groups(family_names, caller, group)

    fig = go.Figure()
    for f, fam in enumerate(family_names):
        in_family = (families == fam).to_numpy()
        member_names = list(pd.unique(members[in_family]))
        for k, member in enumerate(member_names):
            mask = in_family & (members == member).to_numpy()
            base = _style.marker(k, size=_MARKER_SIZE)
            text = _one_column(frame, label).astype(str).to_numpy()[mask] if label else None
            name = member
            fig.add_scatter(
                x=xs[mask], y=ys[mask], mode="markers+text" if label else "markers", name=name or "points",
                legendgroup=fam, marker={**base, "color": _style.color(f) if group else st.accent},
                text=text, textposition="top right", textfont={"size": 10, "color": st.muted},
                showlegend=bool(group),
            )  # fmt: skip
        if connect:
            order = np.argsort(_numeric(frame, connect, caller)[in_family])
            fig.add_scatter(x=xs[in_family][order], y=ys[in_family][order], mode="lines", showlegend=False,
                            hoverinfo="skip", legendgroup=fam,
                            line={"color": _color.with_alpha(_style.color(f), 0.5), "width": 1.4})  # fmt: skip
    fx, fy = xs[front], ys[front]
    order = np.argsort(fx)
    fig.add_scatter(
        x=fx[order],
        y=fy[order],
        mode="lines",
        name="Pareto frontier",
        line={"color": st.muted, "width": 1.4, "dash": "dash"},
    )
    fig.add_scatter(x=fx, y=fy, mode="markers", showlegend=False, hoverinfo="skip",
                    marker={"size": _MARKER_SIZE + 9, "color": "rgba(0,0,0,0)", "line": {"width": 1.5, "color": st.ink_2}})  # fmt: skip
    if logx:
        apply_log(fig, "x", *_log_bounds(xs, caller, "x"))
    if logy:
        apply_log(fig, "y", *_log_bounds(ys, caller, "y"))
    corner = f"{'lower' if lx else 'higher'} {x}, {'lower' if ly else 'higher'} {y} wins"
    return finish(fig, title, _sub(subtitle, f"{corner}; ringed points are the Pareto frontier"),
                  xlabel=xlabel or x, ylabel=ylabel or y, height=height, legend=True)  # fmt: skip


def plot_rank_heatmap(
    frame: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    title: str = "",
    subtitle: str = "",
    *,
    lower_is_better: bool = True,
    height: int | None = None,
) -> go.Figure:
    """Items by contexts, coloured by **rank within each context**. No series cap.

    The one chart that colours by rank. A rank discards magnitude, which is fine for "which model
    is best in this column" and wrong everywhere else (see :func:`plot_correlation`). Rows are
    ordered by mean rank, best first. Hovering shows the underlying value.

    Args:
        frame: Long DataFrame, one row per (item, context).
        index: Column naming the items (rows).
        columns: Column naming the contexts (columns); ranks are computed within each.
        values: Column with the numbers being ranked.
        title: Figure title.
        subtitle: Muted line under the title.
        lower_is_better: Whether a lower value earns the better (lower) rank.
        height: Figure height; grows with the number of rows by default.

    Raises:
        ValueError: duplicated (item, context) pairs.
    """
    caller = "plot_rank_heatmap"
    _require_columns(frame, caller, index=index, columns=columns, values=values)
    wide = frame.pivot(index=index, columns=columns, values=values)
    ranks = wide.rank(method="min", ascending=lower_is_better)
    order = ranks.mean(axis=1).sort_values().index
    wide, ranks = wide.loc[order], ranks.loc[order]
    top = float(np.nanmax(ranks.to_numpy())) if ranks.size else 1.0
    st = _style.get_style()
    ramp = st.sequential
    unit = 1 - (ranks - 1) / max(top - 1, 1)  # best rank = deepest / brightest
    xs, ys = [str(c) for c in ranks.columns], [str(i) for i in ranks.index]
    fig = go.Figure(go.Heatmap(
        z=unit.to_numpy(), x=xs, y=ys, zmin=0, zmax=1, colorscale=_style.sequential_scale(), xgap=1, ygap=1,
        showscale=False, customdata=wide.to_numpy(), hovertemplate="%{y} · %{x}<br>value %{customdata:.4g}<extra></extra>",
    ))  # fmt: skip
    from .matrices import cell_ink  # local import: matrices imports layout, not comparisons

    integral = bool(np.all(np.nan_to_num(ranks.to_numpy()) % 1 == 0))
    rank_grid = ranks.to_numpy(dtype=float)
    unit_grid = unit.to_numpy(dtype=float)
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            rank = float(rank_grid[r, c])
            if np.isnan(rank):
                continue
            fill = _grid_color(ramp, float(unit_grid[r, c]))
            fig.add_annotation(x=x, y=y, showarrow=False, text=f"{rank:.0f}" if integral else f"{rank:.1f}",
                               font={"size": 11, "color": cell_ink(fill), "family": st.mono})  # fmt: skip
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(showgrid=False, type="category")
    sub = _sub(
        subtitle,
        f"rank within each {columns} (1 = best, {'lower' if lower_is_better else 'higher'} {values} wins)",
    )
    return finish(
        fig,
        title,
        sub,
        height=height or int(min(900, max(HEIGHT, 170 + 30 * len(ys)))),
        legend=False,
    )


def _grid_color(ramp: tuple[str, ...], t: float) -> str:
    from .matrices import _ramp_color

    return _ramp_color(ramp, t)


def plot_box_series(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    subtitle: str = "",
    *,
    group: str | None = None,
    label: str | None = None,
    points: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    height: int = HEIGHT,
) -> go.Figure:
    """A box at each numeric ``x``, medians joined so each group reads as a curve.

    The line is the subject and the boxes are its error bars. **``x`` must be numeric, and that is
    not a style choice**: a plotly ``Box`` width is in data units, so the same width renders many
    times wider at one end of a log axis than the other. The x axis is therefore linear, ticked at
    the values that exist. The width comes from the *tightest* gap between two x values divided by
    the group count, so a cluster occupies a fixed share of it whatever the series count. Boxes sit
    at an explicit x with a per-group nudge rather than ``boxmode="group"``, because a grouped box is
    offset from its category centre and the median line would then miss its own boxes.

    Args:
        frame: Long DataFrame, one row per observation.
        x: Numeric column giving where each box sits.
        y: Column with the observations.
        title: Figure title.
        subtitle: Muted line under the title.
        group: Column splitting the observations into curves (at most 8, one per colour).
        label: Column shown on hover for each observation.
        points: Show every observation next to its box.
        xlabel: X axis title; defaults to the column name.
        ylabel: Y axis title; defaults to the column name.
        height: Figure height in px.

    Raises:
        ValueError: ``x`` is not numeric, or more than 8 groups.
    """
    caller = "plot_box_series"
    _require_columns(frame, caller, x=x, y=y, group=group, label=label)
    xs = _numeric(frame, x, caller)
    ys = _numeric(frame, y, caller)
    names = list(pd.unique(_one_column(frame, group).astype(str))) if group else [""]
    if len(names) > MAX_SERIES:
        raise ValueError(
            f"{caller}: group={group!r} has {len(names)} values; at most {MAX_SERIES} (one per colour)"
        )
    positions = np.unique(xs[np.isfinite(xs)])
    gap = float(np.min(np.diff(positions))) if positions.size > 1 else 1.0
    width = gap * 0.8 / len(names)
    step = width * 1.02
    group_col = (
        _one_column(frame, group).astype(str).to_numpy() if group else np.full(len(frame), "")
    )
    fig = go.Figure()
    for i, name in enumerate(names):
        mask = group_col == name
        offset = (i - (len(names) - 1) / 2) * step
        color = _style.color(i)
        fig.add_trace(go.Box(
            x=xs[mask] + offset, y=ys[mask], width=width * 0.92, name=name, legendgroup=name, showlegend=bool(group),
            boxpoints="all" if points else "outliers", pointpos=0, jitter=0.35,
            line={"color": color, "width": 1.4}, fillcolor=_color.with_alpha(color, 0.25),
            marker={"color": color, "size": 4, "opacity": 0.55},
            text=_one_column(frame, label).astype(str).to_numpy()[mask] if label else None,
        ))  # fmt: skip
        med = pd.Series(ys[mask]).groupby(xs[mask]).median().sort_index()
        fig.add_scatter(x=med.index.to_numpy() + offset, y=med.to_numpy(), mode="lines+markers", showlegend=False,
                        legendgroup=name, hoverinfo="skip", line={"color": color, "width": 1.6},
                        marker={"color": color, "size": 5})  # fmt: skip
    fig.update_xaxes(
        type="linear",
        tickmode="array",
        tickvals=positions.tolist(),
        ticktext=[f"{p:g}" for p in positions],
    )
    fig.update_layout(boxmode="overlay")
    return finish(
        fig,
        title,
        subtitle,
        xlabel=xlabel or x,
        ylabel=ylabel or y,
        height=height,
        legend=bool(group),
    )


def plot_small_multiples(
    frame: pd.DataFrame,
    panel: str,
    x: str,
    y: str,
    title: str = "",
    subtitle: str = "",
    *,
    group: str | None = None,
    ncols: int = 3,
    shared_y: bool = True,
    panel_height: int = 240,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> go.Figure:
    """One panel per value of ``panel``, with a shared legend and labels on the outer edges only.

    The answer to "two measures of different scale": one y axis per chart, so give each its own
    panel instead of a second axis.

    Args:
        frame: Long DataFrame.
        panel: Column whose values each get a panel.
        x: Column for the x axis.
        y: Column for the y axis.
        title: Figure title.
        subtitle: Muted line under the title.
        group: Column splitting each panel into coloured lines (at most 3 values).
        ncols: Panels per row.
        shared_y: Share the y range across panels so they are comparable.
        panel_height: Height of one panel row in px.
        xlabel: X axis title; defaults to the column name.
        ylabel: Y axis title; defaults to the column name.

    Raises:
        ValueError: more than 3 groups, or non-numeric ``y``.
    """
    caller = "plot_small_multiples"
    _require_columns(frame, caller, panel=panel, x=x, y=y, group=group)
    _numeric(frame, y, caller)
    panels = [str(p) for p in pd.unique(_one_column(frame, panel))]
    groups = [str(g) for g in pd.unique(_one_column(frame, group))] if group else [""]
    if group:
        _cap_groups(groups, caller, group)
    rows = -(-len(panels) // ncols)
    fig = make_subplots(rows=rows, cols=min(ncols, len(panels)), subplot_titles=panels, shared_yaxes=shared_y,
                        horizontal_spacing=0.06, vertical_spacing=min(0.14, 0.6 / rows))  # fmt: skip
    pcol = _one_column(frame, panel).astype(str).to_numpy()
    gcol = _one_column(frame, group).astype(str).to_numpy() if group else np.full(len(frame), "")
    for p, name in enumerate(panels):
        r, c = p // ncols + 1, p % ncols + 1
        for i, g in enumerate(groups):
            part = frame[(pcol == name) & (gcol == g)].sort_values(x)
            fig.add_scatter(x=part[x], y=part[y], mode="lines+markers", name=g or y, legendgroup=g,
                            showlegend=bool(group) and p == 0, line={"color": _style.color(i), "width": 1.8},
                            marker={"color": _style.color(i), "size": 5}, row=r, col=c)  # fmt: skip
    fig.update_annotations(font={"size": 11, "color": _style.get_style().ink_2})
    fig.update_xaxes(title_text=xlabel or x, row=rows)
    fig.update_yaxes(title_text=ylabel or y, col=1)
    return finish(
        fig, title, subtitle, height=max(HEIGHT, 140 + panel_height * rows), legend=bool(group)
    )


def plot_paired_difference(
    data: SeriesLike,
    base: str,
    title: str = "",
    subtitle: str = "",
    *,
    series: Sequence[str] | None = None,
    xlabel: str = "",
    ylabel: str = "cumulative difference",
    height: int = HEIGHT,
) -> go.Figure:
    """Cumulative difference against a baseline: steady slope, or one cliff?

    This is convention A, but ``base`` is the **name of one of the series** (a dict key for dict
    input), not a column of a long frame. Each other series is compared as ``series - base`` on the
    observations both have, then accumulated.

    Args:
        data: Series, wide DataFrame or ``{label: Series}``.
        base: Name of the baseline series.
        title: Figure title.
        subtitle: Muted line under the title; a default says what is subtracted.
        series: Subset and order of series (must include ``base``).
        xlabel: X axis title.
        ylabel: Y axis title.
        height: Figure height in px.

    Raises:
        KeyError: ``base`` is not one of the series.
        ValueError: fewer than two series, or more than 8.
    """
    frames = pick(as_series_dict(data, _caller="plot_paired_difference"), series)
    if base not in frames:
        raise KeyError(f"base {base!r} not in {list(frames)}; base names a series, not a column")
    others = {k: v for k, v in frames.items() if k != base}
    if not others:
        raise ValueError("plot_paired_difference needs at least one series besides base")
    if len(others) > MAX_SERIES:
        raise ValueError(
            f"plot_paired_difference draws at most {MAX_SERIES} comparisons, got {len(others)}"
        )
    fig = go.Figure()
    fig.add_hline(y=0, line={"color": _style.get_style().axis, "width": 1})
    for i, (name, s) in enumerate(others.items()):
        pair = pd.concat([s, frames[base]], axis=1, join="inner").astype(float).dropna()
        diff = (pair.iloc[:, 0] - pair.iloc[:, 1]).cumsum()
        fig.add_scatter(x=diff.index, y=diff.to_numpy(), mode="lines", name=str(name),
                        line={"color": _style.color(i), "width": 2})  # fmt: skip
    return finish(fig, title, _sub(subtitle, f"each series minus {base}, accumulated"), xlabel=xlabel,
                  ylabel=ylabel, height=height, legend=len(others) > 1)  # fmt: skip


def plot_interval(
    frame: pd.DataFrame,
    low: str,
    high: str,
    point: str,
    title: str = "",
    subtitle: str = "",
    *,
    label: str | None = None,
    group: str | None = None,
    reference: float | None = None,
    sort: bool = True,
    lower_is_better: bool = True,
    xlabel: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """One horizontal interval per row with a marker at ``point``: is the difference bigger than its own uncertainty?

    An interval that crosses ``reference`` is the visual form of *not separable*: it is drawn
    hollow and faded, while separable intervals are solid.

    Args:
        frame: Long DataFrame, one row per interval.
        low: Column with the lower end.
        high: Column with the upper end.
        point: Column with the point estimate.
        title: Figure title.
        subtitle: Muted line under the title.
        label: Column naming each row; defaults to the frame's index.
        group: Column colouring the rows (at most 8 values).
        reference: Value the intervals are compared with (such as 0 for a difference). Draws a
            dotted line and fades intervals that cross it.
        sort: Order rows by ``point``, best at the top.
        lower_is_better: Whether a lower ``point`` is better (decides which end is the top).
        xlabel: X axis title; defaults to ``point``.
        height: Figure height; grows with the number of rows by default.

    Raises:
        ValueError: an interval has ``low > high``, or more than 8 groups.
    """
    caller = "plot_interval"
    _require_columns(frame, caller, low=low, high=high, point=point, label=label, group=group)
    lo, hi, pt = (_numeric(frame, c, caller) for c in (low, high, point))
    if (lo > hi).any():
        raise ValueError(f"{caller}: some rows have {low} > {high}")
    names = (
        _one_column(frame, label).astype(str).to_numpy()
        if label
        else frame.index.astype(str).to_numpy()
    )
    order = np.arange(len(frame))
    if sort:
        order = np.argsort(pt if lower_is_better else -pt, kind="stable")
    st = _style.get_style()
    gcol = _one_column(frame, group).astype(str).to_numpy() if group else np.full(len(frame), "")
    gnames = list(pd.unique(gcol))
    if len(gnames) > MAX_SERIES:
        raise ValueError(
            f"{caller}: group={group!r} has {len(gnames)} values; at most {MAX_SERIES}"
        )
    fig = go.Figure()
    ylabels = [names[i] for i in order]
    for gi, g in enumerate(gnames):
        sel = [i for i in order if gcol[i] == g]
        color = _style.color(gi) if group else st.accent
        for crossing in (False, True):
            rows = (
                [
                    i
                    for i in sel
                    if reference is not None and (lo[i] <= reference <= hi[i]) == crossing
                ]
                if reference is not None
                else (sel if not crossing else [])
            )
            if not rows:
                continue
            fig.add_scatter(
                x=pt[rows], y=[names[i] for i in rows], mode="markers", name=g or "interval",
                showlegend=bool(group) and not crossing, legendgroup=g,
                marker={"size": 9, "color": st.surface if crossing else color, "line": {"width": 2, "color": color}},
                opacity=0.55 if crossing else 1.0,
                error_x={"type": "data", "symmetric": False, "array": (hi[rows] - pt[rows]).tolist(),
                         "arrayminus": (pt[rows] - lo[rows]).tolist(), "color": color, "thickness": 1.8, "width": 4},
                hovertemplate="%{y}<br>%{x:.4g}<extra></extra>",
            )  # fmt: skip
    if reference is not None:
        fig.add_shape(type="line", x0=reference, x1=reference, xref="x", yref="paper", y0=0, y1=1,
                      line={"color": st.muted, "width": 1, "dash": "dot"})  # fmt: skip
    fig.update_yaxes(
        type="category", categoryorder="array", categoryarray=ylabels, autorange="reversed"
    )
    note = (
        "hollow = interval crosses the reference (not separable)" if reference is not None else ""
    )
    return finish(fig, title, _sub(subtitle, note), xlabel=xlabel or point,
                  height=height or int(min(900, max(260, 150 + 30 * len(frame)))), legend=bool(group))  # fmt: skip
