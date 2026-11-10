import numpy as np
import pandas as pd
import pytest

import plotters as pl


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=100)
    return pd.DataFrame(rng.normal(0, 1, (100, 3)), index=idx, columns=list("abc"))


def test_analyze_series_returns_stats_and_five_figures(frame):
    out = pl.analyze_series(frame, "T", bins=20)
    assert list(out["figures"]) == [
        "values",
        "absolute",
        "cumulative",
        "cumulative_absolute",
        "histogram",
    ]
    assert list(out["stats"].index) == ["pooled", "a", "b", "c"]
    assert out["stats"].loc["pooled", "n"] == 300


def test_pooled_distribution_stacks_every_observation(frame):
    out = pl.analyze_series(frame)
    stacked = np.concatenate([frame[c].to_numpy() for c in frame])
    assert out["stats"].loc["pooled", "mean"] == pytest.approx(stacked.mean())
    assert out["stats"].loc["pooled", "std"] == pytest.approx(stacked.std())


def test_pooled_path_averages_available_series_then_accumulates(frame):
    gappy = frame.copy()
    gappy.iloc[0, 0] = np.nan
    out = pl.analyze_series(gappy)
    cumulative = out["figures"]["cumulative"].data[0]
    expected = gappy.mean(axis=1).cumsum()
    assert np.asarray(cumulative.y) == pytest.approx(expected.to_numpy())


def test_absolute_paths_average_magnitudes_before_accumulating(frame):
    out = pl.analyze_series(frame)
    fig = out["figures"]["cumulative_absolute"]
    assert np.asarray(fig.data[0].y) == pytest.approx(frame.abs().mean(axis=1).cumsum().to_numpy())


def test_pooled_false_has_no_pooled_view(frame):
    out = pl.analyze_series(frame, pooled=False)
    assert "pooled" not in out["stats"].index
    assert "pooled" not in [t.name for t in out["figures"]["values"].data]


def test_normalize_in_analysis_scales_each_series(frame):
    out = pl.analyze_series(frame * np.array([1, 10, 100]), normalize="std")
    assert out["stats"].loc[["a", "b", "c"], "std"].to_numpy() == pytest.approx([1, 1, 1])


def test_histogram_denominator_includes_values_outside_range():
    x = pd.Series(np.concatenate([np.zeros(90), np.full(10, 100.0)]), name="x")
    fig = pl.plot_series_distribution(x, bins=10, hist_range=(-1, 1), quantiles=None)
    bar = fig.data[0]
    area = float(np.sum(np.asarray(bar.y) * np.asarray(bar.width)))
    assert area == pytest.approx(0.9)  # 10% of the mass is outside the window and still counted


def test_distribution_has_one_view_per_series_and_a_pooled_view(frame):
    fig = pl.plot_series_distribution(frame, pooled=True)
    assert [b.label for b in fig.layout.updatemenus[0].buttons] == ["pooled", "a", "b", "c"]


def test_normal_reference_is_standard_normal_not_fitted(frame):
    fig = pl.plot_series_distribution(frame["a"], normal_reference=True, quantiles=None)
    line = next(t for t in fig.data if t.name == "N(0, 1)")
    assert max(line.y) == pytest.approx(1 / np.sqrt(2 * np.pi), rel=1e-3)


def test_quantile_markers_use_full_sample():
    x = pd.Series(np.arange(101.0), name="x")
    fig = pl.plot_series_distribution(x, quantiles=[0.5], hist_range=(0, 10))
    assert [s.x0 for s in fig.layout.shapes] == [50.0]


def test_compare_series_difference_is_reference_minus_other(frame):
    out = pl.compare_series(frame, reference="b")
    assert out["difference"]["a"].to_numpy() == pytest.approx((frame["b"] - frame["a"]).to_numpy())
    assert list(out["difference"].columns) == ["a", "c"]


def test_compare_defaults_to_first_series_as_reference(frame):
    assert list(pl.compare_series(frame)["difference"].columns) == ["b", "c"]


def test_compare_align_inner_vs_outer(frame):
    gappy = frame.copy()
    gappy.iloc[5, 0] = np.nan
    gappy.iloc[6, 1] = np.nan
    inner = pl.compare_series(gappy, reference="a", align="inner")
    outer = pl.compare_series(gappy, reference="a", align="outer")
    assert len(inner["data"]) == 98 and len(outer["data"]) == 100
    assert np.isnan(outer["difference"].loc[gappy.index[5], "b"])
    assert not np.isnan(outer["difference"].loc[gappy.index[6], "c"])


def test_compare_validates(frame):
    with pytest.raises(KeyError):
        pl.compare_series(frame, reference="zz")
    with pytest.raises(ValueError):
        pl.compare_series(frame["a"])
    with pytest.raises(ValueError):
        pl.compare_series(frame, align="left")  # type: ignore[arg-type]
