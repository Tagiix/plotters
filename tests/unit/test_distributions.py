import numpy as np
import pandas as pd
import pytest

import plotters as pl


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_null_test_two_sided_p_is_counted_from_the_distribution(rng):
    null = rng.normal(size=100_000)
    fig = pl.plot_null_test(null, 1.96)
    assert "p = 0.0" in fig.layout.annotations[1].text
    p = float(fig.layout.annotations[1].text.split("p = ")[1].split(" ")[0])
    assert p == pytest.approx(0.05, abs=0.006)


def test_null_test_reads_either_way_round(rng):
    boot = rng.normal(loc=3.0, size=50_000)  # the statistic's bootstrap, tested against 0
    assert "reject H0" in pl.plot_null_test(boot, 0.0).layout.annotations[1].text
    assert (
        "cannot reject"
        in pl.plot_null_test(rng.normal(size=50_000), 0.1).layout.annotations[1].text
    )


def test_null_test_one_sided_and_rejection_region_bands(rng):
    null = rng.normal(size=20_000)

    def rects(fig):
        return [sh for sh in fig.layout.shapes if sh.type == "rect"]

    left = rects(pl.plot_null_test(null, -2.0, tail="left"))
    assert len(left) == 1 and left[0].x1 < 0, "only the left tail is shaded"
    assert len(rects(pl.plot_null_test(null, 0.0))) == 2


def test_null_test_validates(rng):
    with pytest.raises(ValueError):
        pl.plot_null_test(rng.normal(size=10), 0.0, alpha=1.5)
    with pytest.raises(ValueError):
        pl.plot_null_test(rng.normal(size=10), 0.0, tail="up")  # type: ignore[arg-type]


def _box_order(fig):
    return [t.name for t in fig.data if t.type == "box"]


def test_box_sorts_by_median_and_first_is_at_the_top():
    df = pd.DataFrame({"hi": [10.0, 11, 12], "lo": [1.0, 2, 3], "mid": [5.0, 6, 7]})
    fig = pl.plot_box(df)
    assert _box_order(fig) == ["lo", "mid", "hi"]
    assert fig.layout.yaxis.autorange == "reversed"
    assert _box_order(pl.plot_box(df, sort="descending")) == ["hi", "mid", "lo"]


def test_box_series_argument_disables_sort_and_uses_one_hue():
    df = pd.DataFrame({"hi": [10.0, 11, 12], "lo": [1.0, 2, 3]})
    fig = pl.plot_box(df, series=["hi", "lo"])
    assert _box_order(fig) == ["hi", "lo"]
    assert len({t.line.color for t in fig.data}) == 1


def test_box_points_log_and_refusals():
    df = pd.DataFrame({"a": [1.0, 2, 3, 100], "b": [2.0, 3, 4, 5]})
    assert all(t.boxpoints == "all" for t in pl.plot_box(df, points=True).data)
    assert pl.plot_box(df, logvalue=True).layout.xaxis.type == "log"
    with pytest.raises(ValueError, match="strictly positive"):
        pl.plot_box(df - 5, logvalue=True)


def test_box_reference_line():
    fig = pl.plot_box(pd.DataFrame({"a": [0.5, 1, 1.5]}), reference=1.0)
    assert fig.layout.shapes[0].x0 == 1.0


def test_histogram_shares_bins_and_density_integrates_to_one(rng):
    df = {"a": pd.Series(rng.normal(0, 1, 5000)), "b": pd.Series(rng.normal(2, 3, 800))}
    fig = pl.plot_histogram(df, density=True, bins=40)
    a, b = fig.data
    assert list(a.x) == list(b.x)
    for t in (a, b):
        assert np.sum(np.asarray(t.y) * np.asarray(t.width)) == pytest.approx(1.0)


def test_histogram_caps_series():
    wide = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 9)))
    with pytest.raises(ValueError, match="at most"):
        pl.plot_histogram(wide)


def test_ecdf_reaches_one_and_complementary_starts_at_one(rng):
    x = pd.Series(rng.normal(size=200))
    assert pl.plot_ecdf(x).data[0].y[-1] == 1.0
    assert pl.plot_ecdf(x, complementary=True).data[0].y[0] == 1.0
    with pytest.raises(ValueError):
        pl.plot_ecdf(x, logx=True)


def test_qq_t_recovers_degrees_of_freedom(rng):
    pytest.importorskip("scipy")
    fig = pl.plot_qq(pd.Series(rng.standard_t(5, 20_000), name="t5"), dist="t")
    nu = float(fig.layout.title.text.split("ν = ")[1].split("<")[0])
    assert nu == pytest.approx(5, abs=0.6)


def test_qq_t_reduces_departure_from_the_reference_on_fat_tails(rng):
    pytest.importorskip("scipy")
    x = pd.Series(rng.standard_t(5, 5000), name="x")

    def departure(fig):
        pts, line = fig.data
        ref = np.interp(pts.x, line.x, line.y)
        return float(np.max(np.abs(np.asarray(pts.y) - ref)))

    assert departure(pl.plot_qq(x, dist="t")) < departure(pl.plot_qq(x, standardize=True))


def test_qq_normal_reference_passes_through_quartiles(rng):
    x = pd.Series(rng.normal(5, 2, 4000), name="x")
    pts, line = pl.plot_qq(x).data
    q = np.quantile(x, [0.25, 0.75])
    at = np.interp([-0.6744897, 0.6744897], line.x, line.y)
    assert at == pytest.approx(q, abs=0.02)


def test_qq_refuses_infinite_variance_t(rng):
    pytest.importorskip("scipy")
    with pytest.raises(ValueError, match="heavy-tailed"):
        pl.plot_qq(pd.Series(rng.standard_cauchy(2000)), dist="t")


def test_cumulative_share_endpoints_and_concentration():
    even = pd.Series([1.0] * 10, name="even")
    skew = pd.Series([100.0] + [1.0] * 9, name="skew")
    fig = pl.plot_cumulative_share(pd.DataFrame({"even": even, "skew": skew}))
    _, e, s = fig.data
    assert (s.x[0], s.y[0], s.x[-1], s.y[-1]) == (0, 0, 1, 1)
    assert s.y[1] > 0.9 > e.y[1]
    with pytest.raises(ValueError, match="negative"):
        pl.plot_cumulative_share(pd.Series([-1.0, 2.0]))


def test_expectile_curve_passes_through_one_and_is_steeper_for_heavier_tails(rng):
    df = pd.DataFrame({"thin": rng.lognormal(0, 0.5, 20_000), "fat": rng.lognormal(0, 1.5, 20_000)})
    fig = pl.plot_expectile_curve(df)
    thin, fat = fig.data
    mid = list(thin.x).index(pytest.approx(0.5))
    assert thin.y[mid] == pytest.approx(1.0, abs=1e-6) and fat.y[mid] == pytest.approx(
        1.0, abs=1e-6
    )
    assert fat.y[-1] > thin.y[-1] > 1
    with pytest.raises(ValueError, match="positive mean"):
        pl.plot_expectile_curve(pd.Series(-df["thin"]))


def test_null_test_observed_label():
    fig = pl.plot_null_test(np.random.default_rng(0).normal(size=300), 0.0, observed_label="equal")
    assert any(a.text == "equal +0" for a in fig.layout.annotations)
