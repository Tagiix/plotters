"""Plots for statistical learning: boundaries, coefficient paths, error curves, ROC, partial
dependence, PCA scree, confusion matrices and dendrograms.

These are the figures a model-fitting workflow keeps coming back to (the kind drawn throughout
*The Elements of Statistical Learning*). They take what you have already computed (a grid of
predictions, a table of coefficients, a frame of errors) and draw it; none of them fits a model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import _color
from . import style as _style
from ._errors import convention_error
from ._types import MAX_ALL_PAIRS, MAX_SERIES, SeriesLike
from .frames import _one_column, _require_columns, as_series_dict, pick
from .layout import HEIGHT, apply_log, finish
from .stats import (
    average_precision,
    one_standard_error_choice,
    precision_recall_curve,
    roc_auc,
    roc_curve,
)

__all__ = [
    "plot_decision_boundary",
    "plot_coefficient_path",
    "plot_error_curves",
    "plot_roc",
    "partial_dependence",
    "plot_partial_dependence",
    "plot_scree",
    "plot_confusion",
    "plot_dendrogram",
]


def _grid_axes(grid: pd.DataFrame, caller: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(grid, pd.DataFrame):
        raise convention_error(caller, "D", grid)
    try:
        return (
            grid.columns.to_numpy(dtype=float),
            grid.index.to_numpy(dtype=float),
            grid.to_numpy(dtype=float),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{caller} needs numeric index (y) and columns (x)") from exc


def plot_decision_boundary(
    grid: pd.DataFrame,
    points: pd.DataFrame | None = None,
    title: str = "",
    subtitle: str = "",
    *,
    x: str | None = None,
    y: str | None = None,
    group: str | None = None,
    level: float = 0.5,
    fill: bool = True,
    reference: pd.DataFrame | None = None,
    reference_label: str = "reference",
    xlabel: str = "",
    ylabel: str = "",
    height: int = 520,
    width: int | None = None,
) -> go.Figure:
    """A classifier's decision boundary over its input plane, with the training points.

    ``grid`` is the model's score evaluated on a regular grid (index = y values, columns = x values),
    so the caller owns the model and this only draws it. The boundary is the contour at ``level``;
    the two sides are tinted lightly so the points stay readable.

    Args:
        grid: Predicted score or probability on a grid (convention D).
        points: Optional long DataFrame of observations to draw on top.
        title: Figure title.
        subtitle: Muted line under the title.
        x: Column of ``points`` for x (required with ``points``).
        y: Column of ``points`` for y (required with ``points``).
        group: Column of ``points`` naming the class; classes are drawn in order of appearance and
            the first is the side *below* ``level``.
        level: Score at which the boundary is drawn (0.5 for probabilities, 0 for margins).
        fill: Tint the two sides.
        reference: A second grid whose boundary is drawn dashed, such as the Bayes-optimal one.
        reference_label: Legend name of the reference boundary.
        xlabel: X axis title.
        ylabel: Y axis title.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ConventionError: ``grid`` is not a DataFrame.
        ValueError: ``points`` without ``x``, ``y`` and ``group``, or more than 3 classes.
    """
    xs, ys, z = _grid_axes(grid, "plot_decision_boundary")
    st = _style.get_style()
    fig = go.Figure()
    if fill:
        below, above = (
            _color.with_alpha(_style.color(0), 0.16),
            _color.with_alpha(_style.color(1), 0.16),
        )
        fig.add_heatmap(x=xs, y=ys, z=(z > level).astype(float), zmin=0, zmax=1, showscale=False, hoverinfo="skip",
                        colorscale=[[0, below], [0.5, below], [0.5, above], [1, above]])  # fmt: skip

    def boundary(values: np.ndarray, name: str, dash: str, width: float, color: str) -> None:
        fig.add_contour(x=xs, y=ys, z=values, showscale=False, hoverinfo="skip", name=name, showlegend=True,
                        contours={"start": level, "end": level, "size": 1, "coloring": "none"},
                        line={"color": color, "width": width, "dash": dash},
                        colorscale=[[0, color], [1, color]])  # fmt: skip

    boundary(z, "boundary", "solid", 2.2, st.ink)
    if reference is not None:
        _, _, ref = _grid_axes(reference, "plot_decision_boundary")
        boundary(ref, reference_label, "dash", 1.6, st.muted)
    if points is not None:
        if x is None or y is None or group is None:
            raise ValueError("plot_decision_boundary: points needs x=, y= and group= column names")
        _require_columns(points, "plot_decision_boundary", x=x, y=y, group=group)
        classes = list(pd.unique(_one_column(points, group)))
        if len(classes) > MAX_ALL_PAIRS:
            raise ValueError(
                f"plot_decision_boundary draws at most {MAX_ALL_PAIRS} classes, got {len(classes)}"
            )
        gcol = _one_column(points, group).to_numpy()
        for i, cls in enumerate(classes):
            m = gcol == cls
            fig.add_scatter(x=points[x].to_numpy()[m], y=points[y].to_numpy()[m], mode="markers", name=str(cls),
                            marker={**_style.marker(i, size=8), "symbol": "circle-open", "line": {"width": 1.8, "color": _style.color(i)}})  # fmt: skip
    fig.update_xaxes(range=[float(xs.min()), float(xs.max())], showgrid=False)
    fig.update_yaxes(range=[float(ys.min()), float(ys.max())], showgrid=False)
    return finish(
        fig,
        title,
        subtitle,
        xlabel=xlabel or (x or ""),
        ylabel=ylabel or (y or ""),
        height=height,
        legend=True,
        width=width,
    )


def plot_coefficient_path(
    paths: pd.DataFrame,
    title: str = "",
    subtitle: str = "",
    *,
    selected: float | None = None,
    logx: bool = False,
    label_ends: bool = True,
    xlabel: str = "shrinkage",
    ylabel: str = "coefficient",
    height: int = HEIGHT,
    width: int | None = None,
) -> go.Figure:
    """How each coefficient moves as the penalty changes: ridge and lasso profiles.

    Index = the shrinkage parameter (λ, effective degrees of freedom, a norm ratio ...), columns =
    coefficients. With many coefficients colours and dashes repeat, so the names are written at the
    right end of each line, nudged apart, instead of in a legend.

    Args:
        paths: DataFrame of coefficients along the path (convention D, numeric index).
        title: Figure title.
        subtitle: Muted line under the title.
        selected: Index value to mark with a vertical line (the chosen penalty).
        logx: Logarithmic x axis (needs a positive index, e.g. λ).
        label_ends: Write coefficient names at the line ends rather than a legend.
        xlabel: X axis title.
        ylabel: Y axis title.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ConventionError: ``paths`` is not a DataFrame.
        ValueError: non-numeric index, or a log axis with non-positive values.
    """
    if not isinstance(paths, pd.DataFrame):
        raise convention_error("plot_coefficient_path", "D", paths)
    try:
        xs = paths.index.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "plot_coefficient_path needs a numeric index (the shrinkage parameter)"
        ) from exc
    st = _style.get_style()
    fig = go.Figure()
    for i, col in enumerate(paths.columns):
        dash, line_width = _style.line_style(i)
        fig.add_scatter(x=xs, y=paths[col].to_numpy(dtype=float), mode="lines", name=str(col),
                        line={"color": _style.color(i), "dash": dash, "width": line_width}, showlegend=not label_ends)  # fmt: skip
    fig.add_hline(y=0, line={"color": st.axis, "width": 1})
    if selected is not None:
        fig.add_shape(type="line", xref="x", x0=selected, x1=selected, yref="paper", y0=0, y1=1,
                      line={"color": st.muted, "width": 1, "dash": "dot"})  # fmt: skip
    if logx:
        if (xs <= 0).any():
            raise ValueError("logx needs strictly positive index values")
        apply_log(fig, "x", float(xs.min()), float(xs.max()))
    fig = finish(
        fig,
        title,
        subtitle,
        xlabel=xlabel,
        ylabel=ylabel,
        height=height,
        legend=not label_ends,
        width=width,
    )
    if label_ends and len(paths.columns):
        ends = paths.iloc[int(np.argmax(xs))].to_numpy(dtype=float)
        order = np.argsort(ends)
        gap = 0.045 * (
            float(np.nanmax(paths.to_numpy())) - float(np.nanmin(paths.to_numpy())) or 1.0
        )
        placed = ends[order].copy()
        for k in range(1, len(placed)):
            placed[k] = max(placed[k], placed[k - 1] + gap)
        for k, col_idx in enumerate(order):
            fig.add_annotation(xref="paper", x=1.0, xanchor="left", xshift=6, yref="y", y=placed[k], showarrow=False,
                               text=str(paths.columns[col_idx]), font={"size": 10, "color": _style.color(int(col_idx))})  # fmt: skip
        fig.update_layout(margin={"r": 110})
    return fig


def plot_error_curves(
    curves: pd.DataFrame,
    title: str = "",
    subtitle: str = "",
    *,
    se: pd.DataFrame | None = None,
    reference: float | Mapping[str, float] | None = None,
    select: str | None = None,
    rule: Literal["min", "1se"] = "min",
    simpler: Literal["low", "high"] = "low",
    logx: bool = False,
    xlabel: str = "model complexity",
    ylabel: str = "error",
    height: int = HEIGHT,
    width: int | None = None,
) -> go.Figure:
    """Training, test and cross-validation error against model complexity, with the chosen model.

    The picture behind bias and variance: training error falls with complexity while test error is
    U-shaped. Index = complexity (k, subset size, λ ...), columns = the curves.

    Args:
        curves: DataFrame of error per complexity (one column per curve).
        title: Figure title.
        subtitle: Muted line under the title.
        se: Standard errors, same shape; drawn as a ribbon behind each curve.
        reference: A horizontal level (or ``{name: level}``), such as the Bayes error.
        select: Name of the curve to choose a complexity on, marked with a ring and a vertical line.
        rule: ``"min"`` takes the minimum; ``"1se"`` takes the simplest model within one standard
            error of it (needs ``se`` for that curve).
        simpler: Which end of the index is the simpler model, for the ``"1se"`` rule.
        logx: Logarithmic x axis.
        xlabel: X axis title.
        ylabel: Y axis title.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ConventionError: ``curves`` is not a DataFrame.
        ValueError: more than ``MAX_SERIES`` curves, an unknown ``select``, or ``"1se"`` without ``se``.
    """
    if not isinstance(curves, pd.DataFrame):
        raise convention_error("plot_error_curves", "D", curves)
    if curves.shape[1] > MAX_SERIES:
        raise ValueError(
            f"plot_error_curves draws at most {MAX_SERIES} curves, got {curves.shape[1]}"
        )
    if rule not in ("min", "1se"):
        raise ValueError(f"rule must be 'min' or '1se', got {rule!r}")
    st = _style.get_style()
    xs = curves.index.to_numpy(dtype=float)
    fig = go.Figure()
    for i, col in enumerate(curves.columns):
        color = _style.color(i)
        y = curves[col].to_numpy(dtype=float)
        if se is not None and col in se.columns:
            e = se[col].to_numpy(dtype=float)
            fig.add_scatter(x=np.r_[xs, xs[::-1]], y=np.r_[y + e, (y - e)[::-1]], fill="toself", mode="lines",
                            line={"width": 0}, fillcolor=_color.with_alpha(color, 0.15), hoverinfo="skip",
                            showlegend=False, legendgroup=str(col))  # fmt: skip
        fig.add_scatter(x=xs, y=y, mode="lines+markers", name=str(col), legendgroup=str(col),
                        line={"color": color, "width": 2}, marker=_style.marker(i, size=5))  # fmt: skip
    levels = (
        {"reference": reference} if isinstance(reference, (int, float)) else dict(reference or {})
    )
    for name, value in levels.items():
        fig.add_hline(y=value, line={"color": st.muted, "width": 1, "dash": "dot"},
                      annotation_text=name, annotation_position="bottom right",
                      annotation_font={"size": 10, "color": st.muted})  # fmt: skip
    note = ""
    if select is not None:
        if select not in curves.columns:
            raise ValueError(f"select={select!r} is not one of {list(curves.columns)}")
        mean = curves[select]
        best: Any = mean.idxmin()
        chosen: Any = best
        if rule == "1se":
            if se is None or select not in se.columns:
                raise ValueError("rule='1se' needs se= with a column for the selected curve")
            chosen = one_standard_error_choice(mean, se[select], simpler=simpler)
            fig.add_scatter(x=[best], y=[mean[best]], mode="markers", showlegend=False, hoverinfo="skip",
                            marker={"size": 13, "color": "rgba(0,0,0,0)", "line": {"width": 1.4, "color": st.muted}})  # fmt: skip
        color = _style.color(list(curves.columns).index(select))
        fig.add_scatter(x=[chosen], y=[mean[chosen]], mode="markers", showlegend=False, hoverinfo="skip",
                        marker={"size": 15, "color": "rgba(0,0,0,0)", "line": {"width": 2, "color": color}})  # fmt: skip
        fig.add_shape(type="line", xref="x", x0=float(chosen), x1=float(chosen), yref="paper", y0=0, y1=1,
                      line={"color": color, "width": 1, "dash": "dot"})  # fmt: skip
        note = (
            f"{select}: {'minimum' if rule == 'min' else 'one-standard-error rule'} at {chosen:g}"
        )
    if logx:
        if (xs <= 0).any():
            raise ValueError("logx needs strictly positive index values")
        apply_log(fig, "x", float(xs.min()), float(xs.max()))
    return finish(
        fig,
        title,
        subtitle or note,
        xlabel=xlabel,
        ylabel=ylabel,
        height=height,
        legend=True,
        width=width,
    )


def plot_roc(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    scores: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    curve: Literal["roc", "pr"] = "roc",
    height: int = 520,
    width: int | None = None,
) -> go.Figure:
    """ROC (or precision-recall) curves for one or several scoring models.

    The legend carries the area under each curve. ROC shows the trade between catching positives and
    raising false alarms; precision-recall is the better lens when positives are rare.

    Args:
        y_true: 0 / 1 labels (1 = positive), in the same order as the scores.
        scores: Higher means more likely positive: a Series, or several models as a wide DataFrame
            or ``{name: Series}``.
        title: Figure title.
        subtitle: Muted line under the title.
        curve: ``"roc"`` or ``"pr"``.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ValueError: labels that are not 0 / 1 or have one class, a length mismatch, an unknown
            ``curve``, or more than ``MAX_SERIES`` models.
    """
    if curve not in ("roc", "pr"):
        raise ValueError(f"curve must be 'roc' or 'pr', got {curve!r}")
    models = pick(as_series_dict(scores, _caller="plot_roc"))
    if len(models) > MAX_SERIES:
        raise ValueError(f"plot_roc draws at most {MAX_SERIES} models, got {len(models)}")
    y = np.asarray(y_true, dtype=float)
    st = _style.get_style()
    fig = go.Figure()
    for i, (name, s) in enumerate(models.items()):
        sv = s.to_numpy(dtype=float)
        if curve == "roc":
            fpr, tpr, _ = roc_curve(y, sv)
            x, yy, area, label = fpr, tpr, roc_auc(y, sv), "AUC"
        else:
            precision, recall, _ = precision_recall_curve(y, sv)
            x, yy, area, label = (
                np.r_[0.0, recall],
                np.r_[precision[0], precision],
                average_precision(y, sv),
                "AP",
            )
        fig.add_scatter(x=x, y=yy, mode="lines", name=f"{name} ({label} {area:.3f})",
                        line={"color": _style.color(i), "width": 2.2, "shape": "hv" if curve == "pr" else "linear"})  # fmt: skip
    if curve == "roc":
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False, hoverinfo="skip",
                        line={"color": st.muted, "width": 1, "dash": "dot"})  # fmt: skip
        fig.update_yaxes(range=[0, 1.01], scaleanchor="x", scaleratio=1, constrain="domain")
        fig.update_xaxes(range=[0, 1], constrain="domain")
        axes = ("false positive rate", "true positive rate")
    else:
        prevalence = float(y.mean())
        fig.add_hline(y=prevalence, line={"color": st.muted, "width": 1, "dash": "dot"})
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1.02])
        axes = ("recall", "precision")
    return finish(
        fig,
        title,
        subtitle,
        xlabel=axes[0],
        ylabel=axes[1],
        height=height,
        legend=True,
        width=width,
    )


def partial_dependence(
    predict: Callable[[pd.DataFrame], np.ndarray],
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    grid_size: int = 20,
    percentiles: tuple[float, float] = (0.05, 0.95),
) -> dict[str, pd.Series]:
    """Average prediction as one feature is swept over a grid and the others keep their data.

    Args:
        predict: Function taking a DataFrame of the same columns as ``frame`` and returning one
            prediction per row.
        frame: The data the model was fitted on (or similar).
        features: Columns to sweep.
        grid_size: Grid points per feature.
        percentiles: The grid spans these quantiles of the feature, so sparse tails do not stretch it.

    Returns:
        ``{feature: Series}`` indexed by the grid values, ready for :func:`plot_partial_dependence`.

    Raises:
        KeyError: a feature is not a column of ``frame``.
    """
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise KeyError(f"{missing} not in {list(frame.columns)}")
    out: dict[str, pd.Series] = {}
    for feature in features:
        lo, hi = (float(frame[feature].quantile(q)) for q in percentiles)
        grid = np.linspace(lo, hi, grid_size)
        values = []
        for g in grid:
            probe = frame.copy()
            probe[feature] = g
            values.append(float(np.mean(predict(probe))))
        out[feature] = pd.Series(values, index=grid, name=feature)
    return out


def plot_partial_dependence(
    curves: Mapping[str, pd.Series],
    data: pd.DataFrame | None = None,
    title: str = "",
    subtitle: str = "",
    *,
    ncols: int = 3,
    shared_y: bool = True,
    panel_height: int = 230,
    height: int | None = None,
    width: int | None = None,
) -> go.Figure:
    """One panel per feature showing how the prediction responds to it.

    Args:
        curves: ``{feature: Series}`` indexed by the grid values (see :func:`partial_dependence`).
        data: Optional data frame; ticks at the deciles of each feature along the bottom show where
            the data actually is, so you do not over-read the sparse ends.
        title: Figure title.
        subtitle: Muted line under the title.
        ncols: Panels per row.
        shared_y: Share the y range so the importance of features can be compared by eye.
        panel_height: Height of one panel row in px.
        height: Total figure height in px; overrides the height computed from ``panel_height``.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ValueError: no curves.
    """
    if not curves:
        raise ValueError("plot_partial_dependence needs at least one curve")
    names = list(curves)
    rows = -(-len(names) // ncols)
    st = _style.get_style()
    fig = make_subplots(rows=rows, cols=min(ncols, len(names)), subplot_titles=names, shared_yaxes=shared_y,
                        horizontal_spacing=0.05, vertical_spacing=min(0.14, 0.6 / rows))  # fmt: skip
    lows = [float(np.min(s.to_numpy())) for s in curves.values()]
    for k, (name, s) in enumerate(curves.items()):
        r, c = k // ncols + 1, k % ncols + 1
        fig.add_scatter(x=s.index.to_numpy(dtype=float), y=s.to_numpy(dtype=float), mode="lines", showlegend=False,
                        line={"color": st.accent, "width": 2.2}, row=r, col=c)  # fmt: skip
        if data is not None and name in data.columns:
            deciles = data[name].quantile(np.linspace(0.1, 0.9, 9)).to_numpy()
            base = min(lows) if shared_y else lows[k]
            fig.add_scatter(x=deciles, y=np.full(deciles.size, base), mode="markers", showlegend=False, hoverinfo="skip",
                            marker={"symbol": "line-ns-open", "size": 9, "color": st.muted}, row=r, col=c)  # fmt: skip
    fig.update_annotations(font={"size": 11, "color": st.ink_2})
    fig.update_yaxes(title_text="partial dependence", col=1)
    return finish(
        fig,
        title,
        subtitle,
        height=height or max(HEIGHT, 120 + panel_height * rows),
        legend=False,
        width=width,
    )


def plot_scree(
    values: Sequence[float] | np.ndarray | pd.Series,
    title: str = "",
    subtitle: str = "",
    *,
    normalize: bool = True,
    cumulative: bool = True,
    threshold: float | None = None,
    top: int | None = None,
    height: int = 420,
    width: int | None = None,
) -> go.Figure:
    """Variance explained by each principal component, with the running total.

    Args:
        values: Eigenvalues (or explained variances), largest first.
        title: Figure title.
        subtitle: Muted line under the title.
        normalize: Convert to shares of the total (so bars and the cumulative line share one axis).
        cumulative: Draw the running total as a line.
        threshold: A cumulative share to mark (for instance 0.9 for "enough components for 90%").
        top: Show only the first ``top`` components.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ValueError: negative values, or ``cumulative`` / ``threshold`` without ``normalize``.
    """
    v = np.asarray(values, dtype=float)
    if (v < 0).any():
        raise ValueError("variances cannot be negative")
    if not normalize and (cumulative or threshold is not None):
        raise ValueError("cumulative and threshold need normalize=True (shares of the total)")
    shares = v / v.sum() if normalize else v
    if top is not None:
        shares = shares[:top]
    st = _style.get_style()
    labels = [f"PC{i + 1}" for i in range(len(shares))]
    fig = go.Figure(go.Bar(x=labels, y=shares, name="variance", marker={"color": _color.with_alpha(st.accent, 0.8), "line": {"width": 0}},
                           hovertemplate="%{x}: %{y:.1%}<extra></extra>" if normalize else None, showlegend=False))  # fmt: skip
    if cumulative:
        fig.add_scatter(x=labels, y=np.cumsum(shares), mode="lines+markers", name="cumulative",
                        line={"color": _style.color(1), "width": 2}, marker=_style.marker(1, size=6))  # fmt: skip
    if threshold is not None:
        fig.add_hline(y=threshold, line={"color": st.muted, "width": 1, "dash": "dot"})
    if normalize:
        fig.update_yaxes(tickformat=".0%", range=[0, 1.02 if cumulative else None])
    return finish(fig, title, subtitle, ylabel="share of variance" if normalize else "variance", height=height,
                  legend=cumulative, width=width)  # fmt: skip


def plot_confusion(
    y_true: Sequence[object] | pd.Series,
    y_pred: Sequence[object] | pd.Series,
    title: str = "",
    subtitle: str = "",
    *,
    labels: Sequence[object] | None = None,
    normalize: Literal["true", "pred", "all"] | None = "true",
    height: int | None = None,
    width: int | None = None,
):
    """Confusion matrix: rows are the actual class, columns the predicted one.

    By default each row is shown as shares of the actual class (so the diagonal is the recall of
    each class and classes of different size stay comparable); hovering shows the counts.

    Args:
        y_true: Actual classes.
        y_pred: Predicted classes, same order.
        title: Figure title.
        subtitle: Muted line under the title.
        labels: Class order for rows and columns; default is sorted order of appearance.
        normalize: ``"true"`` (rows sum to 1), ``"pred"`` (columns sum to 1), ``"all"`` or ``None``
            for raw counts.
        height: Figure height; grows with the number of classes.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ValueError: different lengths or an unknown ``normalize``.
    """
    from .matrices import plot_matrix

    if normalize not in ("true", "pred", "all", None):
        raise ValueError(f"normalize must be 'true', 'pred', 'all' or None, got {normalize!r}")
    a, b = pd.Series(list(y_true)), pd.Series(list(y_pred))
    if len(a) != len(b):
        raise ValueError(f"y_true has {len(a)} values but y_pred has {len(b)}")
    order = list(labels) if labels is not None else sorted(set(a) | set(b), key=str)
    counts = pd.crosstab(a, b).reindex(index=order, columns=order, fill_value=0)
    shown = counts.astype(float)
    if normalize == "true":
        shown = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    elif normalize == "pred":
        shown = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1)
    elif normalize == "all":
        shown = counts / counts.to_numpy().sum()
    fig = plot_matrix(shown.fillna(0.0), title, subtitle or "rows: actual class, columns: predicted class",
                      diverging=False, fmt=".0%" if normalize else ",.0f", xlabel="predicted", ylabel="actual", height=height, width=width)  # fmt: skip
    fig.data[0].update(
        customdata=counts.to_numpy(),
        hovertemplate="actual %{y}, predicted %{x}<br>count %{customdata}<extra></extra>",
    )
    return fig


def plot_dendrogram(
    data: pd.DataFrame,
    title: str = "",
    subtitle: str = "",
    *,
    method: str = "average",
    metric: str = "euclidean",
    n_clusters: int | None = None,
    horizontal: bool = True,
    height: int | None = None,
    width: int | None = None,
):
    """Hierarchical clustering of the rows of a data frame as a tree (needs scipy).

    Branch height is the distance at which two groups merge. A dashed line shows where the tree is
    cut to give ``n_clusters`` groups.

    Args:
        data: Observations in rows, numeric variables in columns; the row labels name the leaves.
        title: Figure title.
        subtitle: Muted line under the title.
        method: Linkage: ``"average"``, ``"complete"``, ``"single"`` or ``"ward"``.
        metric: Distance between observations (any scipy metric).
        n_clusters: Draw the cut that yields this many clusters.
        horizontal: Leaves down the left side (readable for many or long labels); ``False`` puts
            them along the bottom.
        height: Figure height; grows with the number of leaves when horizontal.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ImportError: scipy is not installed.
        ValueError: fewer than 2 rows, or ``n_clusters`` outside 1 to the number of rows.
    """
    try:
        from scipy.cluster import hierarchy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("plot_dendrogram needs scipy: pip install 'plotters[stats]'") from exc
    if not isinstance(data, pd.DataFrame):
        raise convention_error("plot_dendrogram", "D", data)
    n = len(data)
    if n < 2:
        raise ValueError("plot_dendrogram needs at least two rows")
    if n_clusters is not None and not 1 <= n_clusters <= n:
        raise ValueError(f"n_clusters must be between 1 and {n}")
    linkage = hierarchy.linkage(data.to_numpy(dtype=float), method=method, metric=metric)
    tree = hierarchy.dendrogram(linkage, labels=[str(i) for i in data.index], no_plot=True)
    st = _style.get_style()
    leaf, dist, names = np.array(tree["icoord"]), np.array(tree["dcoord"]), tree["ivl"]
    xs: list[float | None] = []
    ys: list[float | None] = []
    for lx, dy in zip(leaf, dist, strict=True):
        xs += [*map(float, lx), None]
        ys += [*map(float, dy), None]
    ticks = [5 + 10 * i for i in range(len(names))]
    fig = go.Figure(go.Scatter(x=(ys if horizontal else xs), y=(xs if horizontal else ys), mode="lines", hoverinfo="skip",
                               line={"color": st.accent, "width": 1.6}, showlegend=False))  # fmt: skip
    if n_clusters is not None and n_clusters > 1:
        heights = linkage[:, 2]
        cut = (
            float((heights[-(n_clusters - 1)] + heights[-n_clusters]) / 2)
            if n_clusters < n
            else 0.0
        )
        fig.add_shape(type="line", line={"color": st.negative, "width": 1.2, "dash": "dash"},
                      **({"x0": cut, "x1": cut, "yref": "paper", "y0": 0, "y1": 1} if horizontal
                         else {"y0": cut, "y1": cut, "xref": "paper", "x0": 0, "x1": 1}))  # fmt: skip
    if horizontal:
        fig.update_yaxes(
            tickmode="array", tickvals=ticks, ticktext=names, showgrid=False, autorange="reversed"
        )
    else:
        fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=names, showgrid=False)
    (fig.update_xaxes if horizontal else fig.update_yaxes)(title_text=f"{method} linkage distance")
    h = height or (int(min(1100, max(300, 130 + 18 * n))) if horizontal else HEIGHT)
    return finish(fig, title, subtitle, height=h, legend=False, width=width)
