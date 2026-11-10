import numpy as np
import pandas as pd
import pytest

import plotters as pl


def test_turnover_ratio_basic():
    s = pd.Series([1.0, 2.0, 1.0, 2.0])
    assert pl.series_turnover_ratio(s) == pytest.approx((1.0) / 1.5)


def test_turnover_ratio_gap_has_no_change_across_it():
    gap = pd.Series([1.0, 1.0, np.nan, 5.0, 5.0])
    assert pl.series_turnover_ratio(gap) == 0.0


def test_turnover_group_level_stays_within_groups():
    idx = pd.MultiIndex.from_product([["a", "b"], [0, 1]])
    s = pd.Series([1.0, 1.0, 9.0, 9.0], index=idx)
    assert pl.series_turnover_ratio(s, group_level=0) == 0.0
    assert pl.series_turnover_ratio(s) > 0


def test_series_statistics_columns_and_ddof():
    x = pd.Series(np.arange(10.0), name="x")
    row = pl.series_statistics(x).loc["x"]
    assert row["std"] == pytest.approx(np.std(np.arange(10.0)))  # ddof=0
    assert row["skewness"] == pytest.approx(0.0, abs=1e-12)
    assert {"n", "mean", "excess_kurtosis", "q50", "tail_2std", "turnover_ratio"} <= set(row.index)


def test_series_statistics_excludes_nan_and_inf():
    x = pd.Series([1.0, 2.0, np.nan, np.inf, 3.0], name="x")
    row = pl.series_statistics(x).loc["x"]
    assert row["n"] == 3 and row["mean"] == pytest.approx(2.0)


def test_thresholds_are_in_input_units_and_metrics_add_columns():
    x = pd.Series([0.0, 1.0, 2.0, 3.0], name="x")
    table = pl.series_statistics(x, thresholds=[1.5], metrics={"rng": lambda s: s.max() - s.min()})
    assert table.loc["x", "tail_1.5"] == 0.5 and table.loc["x", "rng"] == 3.0


def test_normalize_series_std_rms_and_center():
    x = pd.Series([1.0, 2.0, 3.0, 4.0], name="x")
    out = pl.normalize_series(x, "std", center=True)
    assert out.series["x"].mean() == pytest.approx(0) and out.series["x"].std(
        ddof=0
    ) == pytest.approx(1)
    rms = pl.normalize_series(x, "rms")
    assert np.sqrt((rms.series["x"] ** 2).mean()) == pytest.approx(1)
    assert pl.normalise_series is pl.normalize_series


def test_normalize_series_cut_drops_leading_rows():
    x = pd.Series([1.0, 2.0, 3.0, 100.0], index=[1, 2, 3, 0], name="x")
    out = pl.normalize_series(x, None, cut=1)
    assert len(out.series["x"]) == 3
    assert 100.0 not in out.series["x"].to_numpy()  # the leading (earliest-index) row is gone


def test_zero_scale_is_refused():
    with pytest.raises(ValueError, match="zero"):
        pl.normalize_series(pd.Series([2.0, 2.0], name="c"), "std")


def test_expectile_half_is_mean_and_ratio_grows_with_tail_weight():
    rng = np.random.default_rng(1)
    for sigma, expected in ((0.5, 1.75), (1.5, 5.01)):
        x = rng.lognormal(0, sigma, 400_000)
        ratio = pl.sample_expectile(x, 0.95) / x.mean()
        assert ratio == pytest.approx(expected, rel=0.06)
    assert pl.sample_expectile(x, 0.5) == pytest.approx(x.mean(), rel=1e-9)


def test_expectile_validates_tau():
    with pytest.raises(ValueError):
        pl.sample_expectile([1, 2, 3], 1.0)


def test_acf_matches_direct_formula_with_denominator_n():
    rng = np.random.default_rng(2)
    x = rng.normal(size=200)
    d = x - x.mean()
    expected = [np.sum(d[k:] * d[:-k]) / np.sum(d * d) for k in (1, 2, 3)]
    assert pl.acf_values(x, 3) == pytest.approx(expected, abs=1e-15)


def test_acf_rejects_bad_lags_and_constant_series():
    with pytest.raises(ValueError):
        pl.acf_values([1.0, 2.0, 3.0], 3)
    with pytest.raises(ValueError):
        pl.acf_values([1.0] * 10, 2)


def test_bartlett_band_lag_one_is_textbook_and_widens_with_lag():
    rng = np.random.default_rng(3)
    x = np.zeros(1000)
    for t in range(1, 1000):  # AR(1): strong autocorrelation at short lags
        x[t] = 0.7 * x[t - 1] + rng.normal()
    r = pl.acf_values(x, 20)
    band = pl.bartlett_band(r, 1000)
    assert band[0] == pytest.approx(1.959964 / np.sqrt(1000), rel=1e-5)
    assert (np.diff(band) >= -1e-15).all() and band[-1] > band[0] * 1.5
