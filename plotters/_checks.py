"""Small validation helpers shared by the plot modules (private)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ._types import MAX_ALL_PAIRS
from .frames import _one_column

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
