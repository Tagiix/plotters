"""Pure statistics behind the charts: no plotting, plain numbers and DataFrames.

Missing and infinite values are excluded from moments. They also *break* adjacent-difference
statistics (turnover) rather than creating a change across the gap.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
import pandas as pd

from ._types import SeriesLike
from .frames import as_series_dict, pick

__all__ = [
    "Normalized",
    "series_statistics",
    "series_turnover_ratio",
    "normalize_series",
    "normalise_series",
    "sample_expectile",
    "acf_values",
    "bartlett_band",
]


def _f(value: Any) -> float:
    """Plain float from a pandas scalar (the stubs type reductions as ``Scalar``)."""
    return float(value)


def _finite(series: pd.Series) -> pd.Series:
    """``series`` as floats with missing and infinite values removed."""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values[np.isfinite(values)]


def series_turnover_ratio(series: pd.Series, *, group_level: int | str | None = None) -> float:
    """Mean absolute adjacent change divided by mean absolute level.

    A missing or infinite value breaks the differences around it: no change is measured
    across a gap.

    Args:
        series: Observations in order.
        group_level: For a MultiIndex, the level to group by, so differences are taken
            *within* each group and never across the boundary between two.

    Returns:
        The ratio, or ``nan`` when there is nothing to measure.
    """
    x = pd.to_numeric(series, errors="coerce").astype(float)
    x = x.where(np.isfinite(x))
    diff = x.groupby(level=group_level).diff() if group_level is not None else x.diff()
    level = x.abs().mean()
    change = diff.abs().mean()
    if not np.isfinite(level) or level == 0 or not np.isfinite(change):
        return float("nan")
    return float(change / level)


def _quantile_name(q: float) -> str:
    return f"q{q * 100:g}".replace(".", "_")


def _tail_name(t: float) -> str:
    return f"tail_{t:g}"


def series_statistics(
    data: SeriesLike,
    *,
    series: Sequence[str] | None = None,
    quantiles: Sequence[float] = (0.01, 0.05, 0.5, 0.95, 0.99),
    thresholds: Sequence[float] | None = None,
    metrics: Mapping[str, Callable[[pd.Series], float]] | None = None,
    group_level: int | str | None = None,
) -> pd.DataFrame:
    """One row per series: moments, quantiles, tail frequencies and turnover.

    Standard deviation uses ``ddof=0``; skewness and excess kurtosis use pandas' sample
    corrections.

    Args:
        data: Series, wide DataFrame or ``{label: Series}``.
        series: Subset and order of series.
        quantiles: Quantile levels, one column each (``q1``, ``q5``, ``q50`` ...).
        thresholds: Tail cut-offs **in the input's own units**: ``tail_<t>`` is the share of
            observations with ``|x| > t``. ``None`` uses 2 and 3 standard deviations of each
            series (``tail_2std``, ``tail_3std``). No standardisation is implied otherwise.
        metrics: ``{name: callable}``; each callable takes a Series and returns a scalar,
            added as a column.
        group_level: Passed to :func:`series_turnover_ratio`.

    Returns:
        DataFrame indexed by series label with columns ``n, mean, std, skewness,
        excess_kurtosis, min, max``, the quantiles, the tails, ``turnover_ratio`` and any
        custom metrics.
    """
    frames = pick(as_series_dict(data, _caller="series_statistics"), series)
    rows: dict[str, dict[str, float]] = {}
    for name, raw in frames.items():
        x = _finite(raw)
        row: dict[str, float] = {
            "n": float(len(x)),
            "mean": float(x.mean()) if len(x) else np.nan,
            "std": float(x.std(ddof=0)) if len(x) else np.nan,
            "skewness": _f(x.skew()) if len(x) > 2 else np.nan,
            "excess_kurtosis": _f(x.kurt()) if len(x) > 3 else np.nan,
            "min": float(np.min(x)) if len(x) else np.nan,
            "max": float(np.max(x)) if len(x) else np.nan,
        }
        for q in quantiles:
            row[_quantile_name(q)] = float(x.quantile(q)) if len(x) else np.nan
        if thresholds is None:
            sd = row["std"]
            for k in (2, 3):
                row[f"tail_{k}std"] = float((x.abs() > k * sd).mean()) if len(x) else np.nan
        else:
            for t in thresholds:
                row[_tail_name(t)] = float((x.abs() > t).mean()) if len(x) else np.nan
        row["turnover_ratio"] = series_turnover_ratio(raw, group_level=group_level)
        for metric_name, fn in (metrics or {}).items():
            row[metric_name] = float(fn(x))
        rows[str(name)] = row
    return pd.DataFrame.from_dict(rows, orient="index")


@dataclass(frozen=True)
class Normalized:
    """Result of :func:`normalize_series`."""

    series: dict[str, pd.Series]
    means: pd.Series
    scales: pd.Series


def normalize_series(
    data: SeriesLike,
    method: Literal["std", "rms"] | None = "std",
    *,
    center: bool = False,
    ddof: int = 0,
    cut: int = 0,
    series: Sequence[str] | None = None,
) -> Normalized:
    """Scale each series by its own full retained sample: descriptive, not causal.

    Args:
        data: Series, wide DataFrame or ``{label: Series}``.
        method: ``"std"`` divides by the standard deviation, ``"rms"`` by the root mean
            square, ``None`` leaves the scale at 1.
        center: Also subtract each series' mean. The scale is measured *before* centring.
        ddof: Degrees of freedom for ``"std"``.
        cut: Remove this many leading rows after sorting each input's index; the scale and
            mean use only what remains.
        series: Subset and order of series.

    Returns:
        :class:`Normalized` with the transformed ``series`` dict, and the ``means`` and
        ``scales`` that were used.
    """
    if method not in ("std", "rms", None):
        raise ValueError(f"method must be 'std', 'rms' or None, got {method!r}")
    frames = pick(as_series_dict(data, _caller="normalize_series"), series)
    out: dict[str, pd.Series] = {}
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name, raw in frames.items():
        x = raw.sort_index().iloc[cut:].astype(float)
        finite = _finite(x)
        mean = float(finite.mean()) if len(finite) else np.nan
        if method == "std":
            scale = float(finite.std(ddof=ddof))
        elif method == "rms":
            scale = float(np.sqrt((finite**2).mean()))
        else:
            scale = 1.0
        if not np.isfinite(scale) or scale == 0:
            raise ValueError(f"series {name!r} has zero or undefined scale; cannot normalise")
        out[str(name)] = (x - (mean if center else 0.0)) / scale
        means[str(name)] = mean
        scales[str(name)] = scale
    return Normalized(out, pd.Series(means, name="mean"), pd.Series(scales, name="scale"))


normalise_series = normalize_series  # British spelling alias


def sample_expectile(values: Sequence[float] | pd.Series | np.ndarray, tau: float) -> float:
    """The ``tau`` expectile of a sample.

    An expectile is the asymmetric-least-squares analogue of a quantile: unlike a quantile it
    responds to how *large* the exceedances are, not only how many there are. ``tau=0.5`` is
    the mean.

    Args:
        values: Observations; missing and infinite values are dropped.
        tau: Level in (0, 1).

    Raises:
        ValueError: ``tau`` is outside (0, 1) or there are no finite values.
    """
    if not 0 < tau < 1:
        raise ValueError("tau must be strictly between 0 and 1")
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("no finite values")
    lo, hi = float(x.min()), float(x.max())
    if lo == hi:
        return lo
    for _ in range(200):  # bisection on the monotone first-order condition
        mid = (lo + hi) / 2
        d = x - mid
        g = tau * d[d > 0].sum() + (1 - tau) * d[d < 0].sum()
        if g > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, abs(mid)):
            break
    return (lo + hi) / 2


def acf_values(series: pd.Series | np.ndarray | Sequence[float], lags: int) -> np.ndarray:
    """Autocorrelation coefficients at lags 1 to ``lags``, with denominator ``N``.

    This is the biased estimator used by the correlogram convention (so the sequence is a
    valid autocorrelation function).

    Args:
        series: Observations in order; missing and infinite values are dropped first.
        lags: Highest lag.

    Returns:
        Array of length ``lags``; entry ``k - 1`` is the coefficient at lag ``k``.

    Raises:
        ValueError: ``lags`` is below 1 or not smaller than the number of observations.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if lags < 1 or lags >= n:
        raise ValueError(f"lags must be between 1 and {n - 1}")
    d = x - x.mean()
    denom = float((d**2).sum())
    if denom == 0:
        raise ValueError("constant series has no autocorrelation")
    return np.array([(d[k:] * d[:-k]).sum() / denom for k in range(1, lags + 1)])


def bartlett_band(
    coefficients: Sequence[float] | np.ndarray, n: int, alpha: float = 0.05
) -> np.ndarray:
    """Half-width of the significance band at each lag, from Bartlett's formula.

    ``var(r_k) = (1 + 2 * sum_{j<k} r_j**2) / n``: the sampling error at lag ``k`` inherits
    the autocorrelation already present at shorter lags, so the band **widens with the lag**.
    It equals the textbook ``z / sqrt(n)`` only at lag 1; a flat line is wrong elsewhere.

    Args:
        coefficients: Autocorrelations at lags 1 to L (see :func:`acf_values`).
        n: Number of observations the coefficients came from.
        alpha: Two-sided significance level.

    Returns:
        Array of the same length as ``coefficients``.
    """
    r = np.asarray(coefficients, dtype=float)
    z = NormalDist().inv_cdf(1 - alpha / 2)
    cumulative = np.concatenate([[0.0], np.cumsum(r[:-1] ** 2)])
    return z * np.sqrt((1 + 2 * cumulative) / n)
