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


def _axis_values(values: np.ndarray, log: bool) -> np.ndarray:
    """``values`` as drawn: log10 on a log axis (so a centroid sits where the eye puts it)."""
    finite = values[np.isfinite(values)]
    return np.log10(values) if log and finite.size and (finite > 0).all() else values


def _log_bounds(values: np.ndarray, caller: str, axis: str) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or (finite <= 0).any():
        raise ValueError(f"{caller}: log{axis} needs strictly positive values")
    return float(finite.min()) / 1.25, float(finite.max()) * 1.25
