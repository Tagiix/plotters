import math

import numpy as np
import pandas as pd
import pytest

import plotters as pl


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_pacf_matches_yule_walker_and_cuts_off_for_an_ar2(rng):
    x = np.zeros(4000)
    for t in range(2, 4000):
        x[t] = 0.5 * x[t - 1] + 0.3 * x[t - 2] + rng.normal()
    p = pl.pacf_values(x, 6)
    r = np.concatenate([[1.0], pl.acf_values(x, 3)])
    toeplitz = np.array([[r[abs(i - j)] for j in range(3)] for i in range(3)])
    assert p[2] == pytest.approx(np.linalg.solve(toeplitz, r[1:4])[-1])
    assert abs(p[0]) > 0.5 and abs(p[1]) > 0.2 and np.all(np.abs(p[2:]) < 0.08)


def test_pacf_lag_one_equals_acf_lag_one(rng):
    x = rng.normal(size=300).cumsum()
    assert pl.pacf_values(x, 3)[0] == pytest.approx(pl.acf_values(x, 3)[0])


def test_plot_pacf_has_a_flat_band_unlike_the_acf(rng):
    s = pd.Series(rng.normal(size=500), name="x")
    pacf = pl.plot_pacf(s, lags=15)
    upper = next(t for t in pacf.data if t.mode == "lines" and t.line.width == 0)
    assert np.ptp(upper.y) < 1e-12 and upper.y[0] == pytest.approx(
        1.959964 / math.sqrt(500), rel=1e-4
    )
    assert pacf.layout.yaxis.title.text == "pacf(x)"
    grid = pl.plot_acf_grid(pd.DataFrame({"a": s, "b": s[::-1].to_numpy()}), partial=True, lags=10)
    assert grid.layout.yaxis.title.text == "pacf(x)"


def test_ljung_box_detects_autocorrelation_and_accepts_noise(rng):
    pytest.importorskip("scipy")
    noise = rng.normal(size=1000)
    ar = np.zeros(1000)
    for t in range(1, 1000):
        ar[t] = 0.6 * ar[t - 1] + rng.normal()
    assert pl.ljung_box(noise, 10)[1] > 0.01
    assert pl.ljung_box(ar, 10)[1] < 1e-10
    r = pl.acf_values(noise, 5)
    n = 1000
    assert pl.ljung_box(noise, 5)[0] == pytest.approx(
        n * (n + 2) * sum(r[k - 1] ** 2 / (n - k) for k in range(1, 6))
    )


def _history(rng, n=120):
    idx = pd.date_range("2022-01-01", periods=n + 30)
    h = pd.Series(rng.normal(size=n).cumsum(), index=idx[:n], name="y")
    return h, idx[n:]


def test_forecast_draws_history_forecast_interval_and_origin(rng):
    h, fidx = _history(rng)
    f = pd.Series(np.full(30, h.iloc[-1]), index=fidx, name="model")
    fig = pl.plot_forecast(h, f, lower=f - 1, upper=f + 1, actual=f + 0.5)
    assert [t.name for t in fig.data if t.name] == ["history", "model", "actual"]
    assert any(t.fill == "tonexty" for t in fig.data)
    assert fig.layout.shapes[0].x0 == fidx[0], "origin defaults to the first forecast date"
    assert pl.plot_forecast(h, f, split=h.index[-1]).layout.shapes[0].x0 == h.index[-1]


def test_forecast_several_models_and_validation(rng):
    h, fidx = _history(rng)
    two = pd.DataFrame({"a": np.zeros(30), "b": np.ones(30)}, index=fidx)
    fig = pl.plot_forecast(h, two, lower=two - 1, upper=two + 1)
    assert sum(t.fill == "tonexty" for t in fig.data) == 2
    with pytest.raises(ValueError, match="pair"):
        pl.plot_forecast(h, two, lower=two - 1)
    with pytest.raises(KeyError, match="upper has no series"):
        pl.plot_forecast(h, two, lower=two - 1, upper={"a": two["a"]})
    with pytest.raises(pl.ConventionError, match="convention C"):
        pl.plot_forecast(two, two)  # type: ignore[arg-type]


def test_residuals_four_panels_and_default_subtitle(rng):
    r = pd.Series(rng.normal(size=300), index=pd.date_range("2022", periods=300))
    fig = pl.plot_residuals(r, lags=20)
    assert len(fig.layout.annotations) == 4 and "Ljung-Box Q(20)" in fig.layout.title.text
    assert "mean" in fig.layout.title.text and "excess kurtosis" in fig.layout.title.text
    with pytest.raises(ValueError, match="more than"):
        pl.plot_residuals(r.head(10), lags=20)
    with pytest.raises(pl.ConventionError):
        pl.plot_residuals(pd.DataFrame({"a": r}))  # type: ignore[arg-type]


def test_calibration_perfect_forecast_sits_on_the_diagonal(rng):
    p = rng.uniform(0, 1, 20_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    fig = pl.plot_calibration(pd.Series(p, name="m"), y)
    line = next(t for t in fig.data if t.name == "m")
    assert np.max(np.abs(np.asarray(line.x) - np.asarray(line.y))) < 0.04


def test_calibration_detects_overconfidence_and_validates(rng):
    p = rng.uniform(0, 1, 20_000)
    truth = 0.5 + 0.4 * (p - 0.5)  # the real chance is closer to 50% than the forecast says
    y = (rng.uniform(size=p.size) < truth).astype(int)
    fig = pl.plot_calibration(pd.Series(p, name="m"), y, strategy="uniform")
    line = next(t for t in fig.data if t.name == "m")
    assert line.y[0] > line.x[0] + 0.1 and line.y[-1] < line.x[-1] - 0.1
    with pytest.raises(ValueError, match="0 or 1"):
        pl.plot_calibration(pd.Series(p), np.full(p.size, 2))
    with pytest.raises(ValueError, match="outside"):
        pl.plot_calibration(pd.Series(p + 1), y)
    with pytest.raises(ValueError, match="outcomes"):
        pl.plot_calibration(pd.Series(p), y[:10])
    with pytest.raises(ValueError, match="strategy"):
        pl.plot_calibration(pd.Series(p), y, strategy="x")


def test_wilson_interval_stays_inside_zero_one_and_contains_the_estimate():
    from plotters.forecasting import _wilson

    lo, hi = _wilson(np.array([0.0, 5.0, 10.0]), np.array([10.0, 10.0, 10.0]))
    assert (lo >= 0).all() and (hi <= 1).all() and lo[1] < 0.5 < hi[1]


def test_quantile_calibration_honest_and_biased_quantiles(rng):
    n = 3000
    actual = pd.Series(rng.normal(size=n))
    levels = (0.1, 0.5, 0.9)
    honest = pd.DataFrame({t: np.full(n, np.quantile(rng.normal(size=400_000), t)) for t in levels})
    obs = pl.plot_quantile_calibration(actual, honest).data[-1].y
    assert np.allclose(obs, levels, atol=0.03)
    shifted = honest + 1.0
    assert (
        np.asarray(pl.plot_quantile_calibration(actual, shifted).data[-1].y)
        > np.asarray(levels) + 0.05
    ).all()
    with pytest.raises(ValueError, match="between 0 and 1"):
        pl.plot_quantile_calibration(actual, honest.set_axis([0.1, 0.5, 1.5], axis=1))
    with pytest.raises(ValueError, match="numeric levels"):
        pl.plot_quantile_calibration(actual, honest.set_axis(["a", "b", "c"], axis=1))
    two = pl.plot_quantile_calibration(actual, {"m1": honest, "m2": shifted})
    assert [t.name for t in two.data if t.name in ("m1", "m2")] == ["m1", "m2"]


def test_pit_uniform_stays_in_the_band_and_overconfident_forecast_is_u_shaped(rng):
    fig = pl.plot_pit(pd.Series(rng.uniform(size=5000)), bins=10)
    bars = next(t for t in fig.data if t.type == "bar")
    band = fig.data[0]
    outside = (np.asarray(bars.y) < min(band.y)) | (np.asarray(bars.y) > max(band.y))
    assert outside.sum() <= 2, "a pointwise 95% band lets the odd bar of ten escape by chance"
    z = rng.normal(size=5000) * 2  # reality is wider than the unit-variance forecast
    from statistics import NormalDist

    u = pd.Series([NormalDist().cdf(v) for v in z])
    h = next(t for t in pl.plot_pit(u).data if t.type == "bar").y
    assert h[0] > 1.5 and h[-1] > 1.5 and h[4] < 0.8
    with pytest.raises(ValueError, match="outside"):
        pl.plot_pit(pd.Series([0.2, 1.4]))
    assert (
        len(
            [
                t
                for t in pl.plot_pit(pd.DataFrame({"a": u, "b": u})).data
                if t.mode == "lines+markers"
            ]
        )
        == 2
    )


def test_density2d_counts_cells_and_keeps_empty_ones_blank(rng):
    frame = pd.DataFrame({"a": rng.normal(size=20_000), "b": rng.normal(size=20_000)})
    fig = pl.plot_density2d(frame, "a", "b", bins=40, log_counts=False)
    heat = fig.data[0]
    z = np.asarray(heat.customdata, dtype=float)
    assert np.nansum(z) == 20_000 and np.isnan(z).any()
    assert "20,000 points" in fig.layout.title.text
    logged = pl.plot_density2d(frame, "a", "b", bins=40)
    assert logged.data[0].colorbar.ticktext[-1] == f"{np.nanmax(z):.0f}"


def test_density2d_log_axes_and_refusals():
    rng = np.random.default_rng(1)
    pos = pd.DataFrame({"a": rng.lognormal(size=2000), "b": rng.lognormal(size=2000)})
    assert pl.plot_density2d(pos, "a", "b", logx=True, logy=True).layout.xaxis.type == "log"
    with pytest.raises(ValueError, match="strictly positive"):
        pl.plot_density2d(pos - 5, "a", "b", logx=True)
    with pytest.raises(ValueError, match="no finite"):
        pl.plot_density2d(pd.DataFrame({"a": [np.nan], "b": [1.0]}), "a", "b")
    with pytest.raises(pl.ConventionError):
        pl.plot_density2d(pos["a"], "a", "b")  # type: ignore[arg-type]
