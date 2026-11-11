"""Small validation helpers shared by the plot modules (private)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .frames import _one_column

_MARKER_SIZE = 9


def _numeric(frame: pd.DataFrame, column: str, caller: str) -> np.ndarray:
    try:
        return _one_column(frame, column).to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{caller}: column {column!r} must be numeric") from exc


def _label_once(labels: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """``labels`` with every repeat blanked: each value keeps one text, on the point nearest its centroid.

    A label column that repeats (several points per config, say) would otherwise print the same
    name on top of itself once per row.
    """
    out = np.full(labels.shape, "", dtype=object)
    sx, sy = float(np.nanstd(xs)) or 1.0, float(np.nanstd(ys)) or 1.0
    for value in pd.unique(labels):
        idx = np.flatnonzero(labels == value)
        d = ((xs[idx] - np.nanmean(xs[idx])) / sx) ** 2 + (
            (ys[idx] - np.nanmean(ys[idx])) / sy
        ) ** 2
        out[idx[int(np.nanargmin(d))]] = value
    return out


def _declutter(
    labels: np.ndarray, xs: np.ndarray, ys: np.ndarray, priority: np.ndarray, chars_per_axis: float
) -> np.ndarray:
    """``labels`` with every text that would overprint an already kept one blanked.

    Boxes are estimated in axis fractions (text drawn top right of its point); ``priority`` points
    are kept first, then the rest in row order. ``chars_per_axis`` is how many characters fit
    along the x axis.
    """
    keep = np.full(labels.shape, "", dtype=object)
    shown = np.flatnonzero(labels != "")
    if shown.size == 0:
        return keep
    span_x = float(np.nanmax(xs) - np.nanmin(xs)) or 1.0
    span_y = float(np.nanmax(ys) - np.nanmin(ys)) or 1.0
    fx, fy = (xs - np.nanmin(xs)) / span_x, (ys - np.nanmin(ys)) / span_y
    boxes: list[tuple[float, float, float, float]] = []
    for i in shown[np.argsort(~priority[shown], kind="stable")]:
        w = (len(str(labels[i])) + 1) / chars_per_axis
        box = (fx[i], fx[i] + w, fy[i], fy[i] + 0.05)
        if all(box[0] > b[1] or box[1] < b[0] or box[2] > b[3] or box[3] < b[2] for b in boxes):
            boxes.append(box)
            keep[i] = labels[i]
    return keep


def _axis_values(values: np.ndarray, log: bool) -> np.ndarray:
    """``values`` as drawn: log10 on a log axis (so a centroid sits where the eye puts it)."""
    finite = values[np.isfinite(values)]
    return np.log10(values) if log and finite.size and (finite > 0).all() else values


def _log_bounds(values: np.ndarray, caller: str, axis: str) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or (finite <= 0).any():
        raise ValueError(f"{caller}: log{axis} needs strictly positive values")
    return float(finite.min()) / 1.25, float(finite.max()) * 1.25
