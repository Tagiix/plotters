"""Data shaping: the bridge between long frames and the series-like convention.

Every plotter normalises its input through this module first, so chart code never needs to
know whether it was handed a Series, a DataFrame or a dict.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from ._errors import convention_error
from ._types import SeriesLike

__all__ = [
    "as_series_dict",
    "pick",
    "long_to_wide",
    "group_values",
    "normalize_within",
    "normalise_within",
    "group_by_prefix",
    "label_from",
    "as_value_series",
]


def as_series_dict(
    data: SeriesLike, value: str = "value", *, _caller: str = "this plotter"
) -> dict[str, pd.Series]:
    """Normalise convention-A input to ``{label: Series}``.

    A Series becomes one series named from ``.name``; a DataFrame becomes one series per
    column (so it must be wide, not tidy); a dict is used as is.

    Args:
        data: Series, wide DataFrame or ``{label: Series}``.
        value: Label for an unnamed Series.

    Returns:
        A new dict, in input order. Labels are converted to ``str``.

    Raises:
        ConventionError: ``data`` is none of the three shapes.
        ValueError: a DataFrame has duplicated column labels.
    """
    if isinstance(data, pd.Series):
        return {str(data.name) if data.name is not None else value: data}
    if isinstance(data, pd.DataFrame):
        if data.columns.duplicated().any():
            dup = list(data.columns[data.columns.duplicated()].unique())
            raise ValueError(f"duplicated column labels {dup}: series would be ambiguous")
        return {str(col): _one_column(data, col) for col in data.columns}
    if isinstance(data, Mapping):
        out: dict[str, pd.Series] = {}
        for key, val in data.items():
            if not isinstance(val, pd.Series):
                raise convention_error(
                    _caller, "A", val, f"The value for {key!r} must be a Series."
                )
            out[str(key)] = val
        return out
    raise convention_error(_caller, "A", data)


def as_value_series(data: Mapping[str, float] | pd.Series, name: str = "value") -> pd.Series:
    """Normalise convention-E input (label to number) to a float Series indexed by ``str`` labels.

    Args:
        data: A ``{label: number}`` dict, or a Series indexed by label (for instance the result of
            ``value_counts()``).
        name: Name given to the resulting Series.

    Raises:
        ConventionError: ``data`` is neither a Series nor a mapping (a DataFrame is refused too:
            it is a table, not a label-to-number mapping).
        ValueError: a value is not numeric.
    """
    if isinstance(data, pd.Series):
        series = data.copy()
    elif isinstance(data, Mapping):
        series = pd.Series(dict(data))
    else:
        raise convention_error("this plotter", "E", data)
    try:
        series = series.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("label-to-number input must have numeric values") from exc
    series.index = series.index.map(str)
    series.name = name if series.name is None else str(series.name)
    return series


def pick(
    series: Mapping[str, pd.Series], keep: Sequence[str] | None = None
) -> dict[str, pd.Series]:
    """Subset a series dict, **in the order given**; this is what ``series=`` uses.

    Args:
        series: ``{label: Series}``.
        keep: Labels to keep, in the order you want them (and so their colour slots).
            ``None`` keeps everything in the dict's order.

    Raises:
        KeyError: a label in ``keep`` is not in ``series``, listing what is available.
    """
    if keep is None:
        return dict(series)
    missing = [k for k in keep if k not in series]
    if missing:
        raise KeyError(f"{missing} not in {list(series)}")
    return {k: series[k] for k in keep}


def _numeric_order(labels: Iterable[object]) -> list[object] | None:
    """Labels sorted numerically if all of them parse as numbers, else ``None``."""
    labels = list(labels)
    try:
        keys = [float(label) for label in labels]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return [label for _, label in sorted(zip(keys, labels, strict=True), key=lambda p: p[0])]


def long_to_wide(frame: pd.DataFrame, index: str, columns: str, values: str) -> pd.DataFrame:
    """Pivot a long frame to wide, ready for convention A.

    Numeric-looking columns are ordered **numerically**. ``pivot`` sorts string labels as
    strings, so ``126`` would land between ``18`` and ``21`` and every line chart would zigzag.

    Args:
        frame: Long DataFrame, one row per observation.
        index: Column that becomes the row index (usually time).
        columns: Column whose values become the series.
        values: Column holding the numbers.

    Raises:
        KeyError: a named column is not in ``frame``.
        ValueError: duplicated ``(index, columns)`` pairs (``pivot`` cannot choose).
    """
    _require_columns(frame, "long_to_wide", index=index, columns=columns, values=values)
    wide = frame.pivot(index=index, columns=columns, values=values)
    order = _numeric_order(wide.columns)
    return wide[order] if order is not None else wide


def group_values(
    frame: pd.DataFrame, by: str, value: str, *, keep: Sequence[str] | None = None
) -> dict[str, pd.Series]:
    """Split a long frame into ``{group: Series}``, for the distribution plotters.

    Args:
        frame: Long DataFrame.
        by: Column naming the group.
        value: Column with the observations.
        keep: Group names to keep, in the order you want. ``None`` orders by median.

    Raises:
        KeyError: a column is missing, or ``keep`` names an unknown group.
    """
    _require_columns(frame, "group_values", by=by, value=value)
    groups = {
        str(name): pd.Series(part[value].to_numpy(), name=str(name))
        for name, part in frame.groupby(by, sort=False)
    }
    if keep is not None:
        return pick(groups, keep)
    return dict(sorted(groups.items(), key=lambda kv: kv[1].median()))


_REFERENCES = {
    "min": "min",
    "max": "max",
    "mean": "mean",
    "median": "median",
    "first": "first",
}


def normalize_within(
    frame: pd.DataFrame,
    value: str,
    within: str,
    how: Literal["min", "max", "mean", "median", "first"] = "min",
) -> pd.Series:
    """Divide ``value`` by its own group's reference; the fix for pooling across levels.

    Args:
        frame: Long DataFrame.
        value: Column to rescale.
        within: Column naming the group whose reference is used.
        how: The reference: ``"min"``, ``"max"``, ``"mean"``, ``"median"`` or ``"first"``.

    Returns:
        A Series aligned to ``frame``'s index.
    """
    _require_columns(frame, "normalize_within", value=value, within=within)
    if how not in _REFERENCES:
        raise ValueError(f"how must be one of {sorted(_REFERENCES)}, got {how!r}")
    per_group = getattr(frame.groupby(within)[value], _REFERENCES[how])()
    return pd.Series(frame[value] / frame[within].map(per_group), name=value)


normalise_within = normalize_within  # British spelling alias


def group_by_prefix(
    labels: Iterable[str],
    groups: Mapping[str, Sequence[str]],
    *,
    other: str | None = None,
) -> pd.Series:
    """Map each label to a group name by its **longest** matching prefix, for ``group=``.

    Longest wins, so with ``{"ewrms": ("ewrms",), "range": ("ewrms_gk",)}`` the label
    ``ewrms_gk`` lands in ``range`` and the dict's order cannot decide it.

    Args:
        labels: Labels to classify (a Series keeps its index).
        groups: ``{group: (prefix, ...)}``.
        other: Group for labels matching nothing. ``None`` raises instead: a silent
            bucket hides the entry you forgot.

    Raises:
        ValueError: a label matches no prefix and ``other`` is ``None``.
    """
    index = labels.index if isinstance(labels, pd.Series) else None
    names = [str(label) for label in labels]
    table = sorted(
        ((prefix, group) for group, prefixes in groups.items() for prefix in prefixes),
        key=lambda pair: -len(pair[0]),
    )
    out: list[str] = []
    unmatched: list[str] = []
    for name in names:
        hit = next((group for prefix, group in table if name.startswith(prefix)), None)
        if hit is None:
            if other is None:
                unmatched.append(name)
                continue
            hit = other
        out.append(hit)
    if unmatched:
        raise ValueError(
            f"labels match no prefix: {sorted(set(unmatched))}; pass other= to bucket them"
        )
    return pd.Series(out, index=index, name="group")


def label_from(frame: pd.DataFrame, *columns: str, sep: str = " ") -> pd.Series:
    """Build a label column from several columns, for ``label=``; NaN parts are skipped.

    A row with no parameter reads ``model``, not ``model nan``.

    Args:
        frame: DataFrame.
        *columns: Columns to join, left to right.
        sep: Separator.
    """
    _require_columns(frame, "label_from", **{f"column {i}": c for i, c in enumerate(columns)})
    parts = frame[list(columns)].astype(object)

    def join(row: pd.Series) -> str:
        return sep.join(str(v) for v in row if pd.notna(v) and str(v) != "")

    return pd.Series(parts.apply(join, axis=1), index=frame.index, name="label")


def _require_columns(frame: object, caller: str, **named: str | None) -> None:
    """Check that ``frame`` is a DataFrame and every named column exists.

    Raises:
        ConventionError: ``frame`` is not a DataFrame.
        KeyError: a column is missing, listing the columns available.
    """
    if not isinstance(frame, pd.DataFrame):
        raise convention_error(caller, "B", frame)
    missing = {
        arg: col for arg, col in named.items() if col is not None and col not in frame.columns
    }
    if missing:
        raise KeyError(f"{caller}: {missing} not in columns {list(frame.columns)}")


def _one_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return ``frame[name]`` as a Series, raising if the label is duplicated.

    ``frame[label]`` is typed ``Series | DataFrame``; this narrows it once. A duplicated
    column label used to produce a wrong chart silently.
    """
    column = frame[name]
    if isinstance(column, pd.DataFrame):
        raise ValueError(f"column label {name!r} is duplicated; rename one of them")
    return column


def _as_array(values: pd.Series | np.ndarray) -> np.ndarray:
    """Plain float array of a Series, for passing to plotly."""
    return np.asarray(values, dtype=float)
