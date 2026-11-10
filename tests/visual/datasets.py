"""Seeded synthetic data for the gallery. Same numbers on every run."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def dates(n: int = 400) -> pd.DatetimeIndex:
    return pd.date_range("2022-01-03", periods=n, freq="B")


def levels(k: int = 3, n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Random-walk levels around 100, columns ``model_0 ...``."""
    steps = _rng(seed).normal(0, 1, (n, k))
    return pd.DataFrame(
        100 + steps.cumsum(0), index=dates(n), columns=[f"model_{i}" for i in range(k)]
    )


def returns(k: int = 3, n: int = 400, seed: int = 1) -> pd.DataFrame:
    """Daily returns as fractions (0.01 is 1%), mild fat tails."""
    r = _rng(seed).standard_t(5, (n, k)) * 0.008
    return pd.DataFrame(
        r,
        index=dates(n),
        columns=["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta"][:k],
    )


def positive_growth(n: int = 300) -> pd.Series:
    """Strictly positive series that crosses 1, for log-axis cards."""
    return pd.Series(
        np.exp(np.linspace(-1.4, 1.6, n) + _rng(3).normal(0, 0.04, n)),
        index=dates(n),
        name="growth",
    )


def shares(k: int = 4, n: int = 120) -> pd.DataFrame:
    raw = np.abs(_rng(4).normal(1, 0.3, (n, k))).cumsum(0) / np.arange(1, n + 1)[:, None] + 1
    return pd.DataFrame(raw, index=dates(n), columns=[f"part_{i}" for i in range(k)])
