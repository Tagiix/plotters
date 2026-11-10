import numpy as np
import pandas as pd
import pytest

import plotters as pl


def ar1(n=800, phi=0.6, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return pd.Series(x, name="ar")


def test_single_series_has_stems_markers_and_band():
    fig = pl.plot_acf(ar1(), lags=20)
    markers = next(t for t in fig.data if t.mode == "markers")
    assert len(markers.y) == 20
    assert any(t.fill == "tonexty" for t in fig.data), "the band is a ribbon"


def test_band_widens_with_lag_and_matches_textbook_at_lag_one():
    fig = pl.plot_acf(ar1(), lags=20)
    upper = next(t for t in fig.data if t.mode == "lines" and t.line.width == 0)
    assert upper.y[0] == pytest.approx(1.959964 / np.sqrt(800), rel=1e-4)
    assert upper.y[-1] > upper.y[0] * 1.3


def test_multiple_series_have_no_shared_band_and_are_offset():
    df = pd.DataFrame({"a": ar1(), "b": ar1(seed=1)})
    fig = pl.plot_acf(df, lags=10)
    assert not any(t.fill == "tonexty" for t in fig.data)
    marks = [t for t in fig.data if t.mode == "markers"]
    assert marks[0].x[0] != marks[1].x[0]


def test_acf_draws_many_series():
    wide = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 12)))
    assert len(pl.plot_acf(wide).data) >= 12


def test_grid_has_a_panel_and_its_own_band_per_series_with_no_cap():
    wide = pd.DataFrame(
        np.random.default_rng(0).normal(size=(300, 11)), columns=[f"s{i}" for i in range(11)]
    )
    fig = pl.plot_acf_grid(wide, lags=15, ncols=3)
    bands = [t for t in fig.data if t.fill == "tonexty"]
    assert len(bands) == 11
    assert len({tuple(np.round(b.y, 12)) for b in bands}) == 11, "each band is its own"


def test_absolute_detects_volatility_clustering_that_signed_misses():
    rng = np.random.default_rng(4)
    sigma = np.ones(2000)
    z = rng.normal(size=2000)
    x = np.zeros(2000)
    for t in range(1, 2000):  # ARCH(1): uncorrelated signs, clustered magnitude
        sigma[t] = np.sqrt(0.2 + 0.7 * x[t - 1] ** 2)
        x[t] = sigma[t] * z[t]
    s = pd.Series(x, name="arch")
    signed = np.max(np.abs(pl.acf_values(s, 10)))
    magnitude = np.max(pl.acf_values(s.abs(), 10))
    assert magnitude > 3 * signed
    fig = pl.plot_acf(s, absolute=True, lags=10)
    assert fig.layout.yaxis.title.text == "acf(|x|)"
