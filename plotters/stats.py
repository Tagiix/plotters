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
    "pacf_values",
    "ljung_box",
    "one_standard_error_choice",
    "roc_curve",
    "roc_auc",
    "precision_recall_curve",
    "average_precision",
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


def pacf_values(series: pd.Series | np.ndarray | Sequence[float], lags: int) -> np.ndarray:
    """Partial autocorrelation at lags 1 to ``lags``, by the Durbin-Levinson recursion.

    The partial autocorrelation at lag ``k`` is the correlation between ``x[t]`` and ``x[t-k]``
    after removing what the lags in between explain. For an AR(p) series it cuts off after lag p,
    where the plain acf only decays.

    Args:
        series: Observations in order; missing and infinite values are dropped first.
        lags: Highest lag.

    Returns:
        Array of length ``lags``; entry ``k - 1`` is the partial autocorrelation at lag ``k``.

    Raises:
        ValueError: ``lags`` is below 1 or not smaller than the number of observations.
    """
    r = np.concatenate([[1.0], acf_values(series, lags)])
    phi = np.zeros((lags + 1, lags + 1))
    out = np.zeros(lags)
    phi[1, 1] = out[0] = r[1]
    for k in range(2, lags + 1):
        num = r[k] - phi[k - 1, 1:k] @ r[k - 1 : 0 : -1]
        den = 1.0 - phi[k - 1, 1:k] @ r[1:k]
        phi[k, k] = out[k - 1] = num / den
        phi[k, 1:k] = phi[k - 1, 1:k] - phi[k, k] * phi[k - 1, k - 1 : 0 : -1]
    return out


def ljung_box(
    series: pd.Series | np.ndarray | Sequence[float], lags: int, *, model_df: int = 0
) -> tuple[float, float]:
    """Ljung-Box portmanteau test that the first ``lags`` autocorrelations are jointly zero.

    ``Q = n (n + 2) * sum_k r_k**2 / (n - k)``, compared with a chi-square on ``lags - model_df``
    degrees of freedom. The p value needs scipy (imported lazily); without it ``nan`` is returned
    for p and Q is still valid.

    Args:
        series: Residuals or any series; missing and infinite values are dropped first.
        lags: Number of autocorrelations tested.
        model_df: Parameters fitted by the model that produced the residuals, subtracted from the
            degrees of freedom.

    Returns:
        ``(Q, p)``.
    """
    x = np.asarray(series, dtype=float)
    n = int(np.isfinite(x).sum())
    r = acf_values(x, lags)
    q = float(n * (n + 2) * np.sum(r**2 / (n - np.arange(1, lags + 1))))
    try:
        from scipy import stats as _sps  # type: ignore[import-not-found]
    except ImportError:
        return q, float("nan")
    dof = max(lags - model_df, 1)
    return q, float(_sps.chi2.sf(q, dof))


def one_standard_error_choice(
    mean: pd.Series, se: pd.Series, *, simpler: Literal["low", "high"] = "low"
) -> object:
    """The one-standard-error rule: the simplest model within one standard error of the best.

    Cross-validated error is noisy, so the minimum is partly luck. The rule takes the simplest
    candidate whose error is no more than one standard error above the minimum's.

    Args:
        mean: Error per candidate, indexed by complexity (smaller is better).
        se: Standard error of that error, same index.
        simpler: Which end of the index is the simpler model: ``"low"`` (fewer terms, smaller k...)
            or ``"high"`` (a larger penalty, a bigger k in k-NN).

    Returns:
        The index label of the chosen candidate.

    Raises:
        ValueError: the indexes differ or are empty.
    """
    if not mean.index.equals(se.index) or mean.empty:
        raise ValueError("mean and se need the same, non-empty index")
    best: Any = mean.idxmin()
    limit = float(mean[best] + se[best])
    eligible = mean[mean <= limit].index
    ordered = sorted(eligible)
    return ordered[0] if simpler == "low" else ordered[-1]


def _binary(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    score: Sequence[float] | np.ndarray | pd.Series,
):
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    if y.shape != s.shape:
        raise ValueError(f"y_true has {y.size} values but score has {s.size}")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_true must be 0 or 1")
    if y.sum() in (0, y.size):
        raise ValueError("y_true needs both classes")
    ok = np.isfinite(s)
    return y[ok], s[ok]


def _cumulative_counts(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    last = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]  # one point per distinct threshold
    return tps[last], fps[last], ss[last]


def roc_curve(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    score: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """False-positive and true-positive rates for every distinct score threshold.

    Args:
        y_true: 0 / 1 labels (1 = positive).
        score: Higher means more likely positive.

    Returns:
        ``(fpr, tpr, thresholds)``, starting at ``(0, 0)`` (threshold ``inf``).
    """
    y, s = _binary(y_true, score)
    tps, fps, thr = _cumulative_counts(y, s)
    return (
        np.r_[0.0, fps / (1 - y).sum()],
        np.r_[0.0, tps / y.sum()],
        np.r_[np.inf, thr],
    )


def roc_auc(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    score: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Area under the ROC curve: the chance a random positive outscores a random negative."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def precision_recall_curve(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    score: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision and recall for every distinct score threshold.

    Returns:
        ``(precision, recall, thresholds)``, in decreasing-threshold order.
    """
    y, s = _binary(y_true, score)
    tps, fps, thr = _cumulative_counts(y, s)
    return tps / (tps + fps), tps / y.sum(), thr


def average_precision(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    score: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Area under the precision-recall curve as a step sum: ``sum (R_k - R_{k-1}) * P_k``."""
    precision, recall, _ = precision_recall_curve(y_true, score)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
