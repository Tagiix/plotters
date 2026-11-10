"""Gallery cards for forecast evaluation, calibration and the larger-data scatter."""

from statistics import NormalDist

import numpy as np
import pandas as pd

import plotters as pl

from .registry import card


def _series(n=160, seed=50):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n + 30, freq="B")
    level = 100 + np.cumsum(rng.normal(0.1, 1, n + 30))
    return pd.Series(level, index=idx), n


@card(
    "Forecasting",
    "plot_forecast",
    [
        "History in dark ink, the forecast in colour with a translucent interval, the realised path dotted.",
        "A dotted vertical line marks where the forecast starts.",
    ],
)
def fc_forecast():
    full, n = _series()
    hist, future = full.iloc[:n], full.iloc[n:]
    horizon = np.arange(1, len(future) + 1)
    point = pd.Series(hist.iloc[-1] + 0.1 * horizon, index=future.index, name="drift model")
    width = 1.0 * np.sqrt(horizon) * 1.96
    return pl.plot_forecast(
        hist,
        point,
        "Forecast with a 95% interval",
        lower=point - width,
        upper=point + width,
        actual=future.rename("actual"),
    )


@card(
    "Forecasting",
    "plot_forecast: two models",
    [
        "Two forecasts over the same horizon, each with its own interval ribbon in its own colour.",
    ],
)
def fc_forecast_two():
    full, n = _series(seed=51)
    hist, future = full.iloc[:n], full.iloc[n:]
    h = np.arange(1, len(future) + 1)
    a = pd.Series(hist.iloc[-1] + 0.1 * h, index=future.index)
    b = pd.Series(hist.iloc[-1] + 0.02 * h, index=future.index)
    models = pd.DataFrame({"trend": a, "flat": b})
    w = pd.DataFrame({"trend": 1.5 * np.sqrt(h), "flat": 1.0 * np.sqrt(h)}, index=future.index)
    return pl.plot_forecast(
        hist,
        models,
        "Comparing two forecasts",
        lower=models - w,
        upper=models + w,
        actual=future.rename("actual"),
    )


@card(
    "Forecasting",
    "plot_residuals",
    [
        "Four panels: residuals over time, histogram with a normal of the same spread, normal QQ, and the acf with its band.",
        "The subtitle reports mean, std, skew, excess kurtosis and the Ljung-Box test; this series has leftover autocorrelation, so p is tiny.",
    ],
)
def fc_residuals():
    rng = np.random.default_rng(52)
    e = np.zeros(400)
    for t in range(1, 400):
        e[t] = 0.35 * e[t - 1] + rng.standard_t(6)
    return pl.plot_residuals(
        pd.Series(e, index=pd.date_range("2022-01-03", periods=400, freq="B")),
        "Model residuals",
        lags=25,
    )


@card(
    "Forecasting",
    "plot_residuals: against fitted",
    [
        "First panel plots residuals against the fitted value: the funnel shows the spread growing with the level.",
    ],
)
def fc_residuals_fitted():
    rng = np.random.default_rng(53)
    fitted = pd.Series(rng.uniform(1, 10, 400), index=pd.RangeIndex(400))
    resid = pd.Series(rng.normal(0, 0.3 * fitted.to_numpy()), index=fitted.index)
    return pl.plot_residuals(resid, "Heteroskedastic residuals", fitted=fitted, lags=20)


@card(
    "Correlogram",
    "plot_pacf",
    [
        "AR(2) series: the pacf cuts off after lag 2 (compare the slowly decaying acf).",
        "The band is flat (z over root n), unlike the acf's widening Bartlett band.",
    ],
)
def fc_pacf():
    rng = np.random.default_rng(54)
    x = np.zeros(1500)
    for t in range(2, 1500):
        x[t] = 0.5 * x[t - 1] + 0.3 * x[t - 2] + rng.normal()
    return pl.plot_pacf(pd.Series(x, name="AR(2)"), "Partial autocorrelation", lags=20)


@card(
    "Correlogram",
    "plot_acf_grid: partial",
    [
        "partial=True: one pacf panel per series, each with its own flat band.",
    ],
)
def fc_pacf_grid():
    rng = np.random.default_rng(55)
    ar = np.zeros(800)
    for t in range(1, 800):
        ar[t] = 0.7 * ar[t - 1] + rng.normal()
    return pl.plot_acf_grid(
        pd.DataFrame({"AR(1)": ar, "noise": rng.normal(size=800)}),
        "Pacf per series",
        lags=15,
        partial=True,
    )


@card(
    "Forecasting",
    "plot_calibration",
    [
        "Reliability diagram: the honest model hugs the dotted diagonal; the overconfident one is flatter than it (above it at the low end, below it at the high end).",
        "Vertical bars are 95% Wilson intervals.",
    ],
)
def fc_calibration():
    rng = np.random.default_rng(56)
    truth = rng.uniform(0.02, 0.98, 6000)
    y = (rng.uniform(size=6000) < truth).astype(int)
    overconfident = 1 / (1 + np.exp(-2.2 * np.log(truth / (1 - truth))))
    return pl.plot_calibration(
        pd.DataFrame({"honest": truth, "overconfident": overconfident}),
        y,
        "Are the probabilities honest?",
        bins=10,
    )


@card(
    "Forecasting",
    "plot_quantile_calibration",
    [
        "Observed fraction below each forecast quantile against its nominal level; the shaded ribbon is what noise alone allows.",
        "The narrow model's points leave the ribbon at both ends: its intervals are too tight.",
    ],
)
def fc_qcal():
    rng = np.random.default_rng(57)
    n = 1500
    actual = pd.Series(rng.normal(0, 1, n))
    levels = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    z = {t: NormalDist().inv_cdf(t) for t in levels}
    right = pd.DataFrame({t: np.full(n, z[t]) for t in levels})
    narrow = pd.DataFrame({t: np.full(n, 0.6 * z[t]) for t in levels})
    return pl.plot_quantile_calibration(
        actual, {"right spread": right, "too narrow": narrow}, "Quantile calibration"
    )


@card(
    "Forecasting",
    "plot_pit",
    [
        "A flat histogram inside the ribbon is calibrated; the U shape is an overconfident forecast distribution.",
    ],
)
def fc_pit():
    rng = np.random.default_rng(58)
    z = rng.normal(0, 1, 3000)
    nd = NormalDist()
    good = pd.Series([nd.cdf(v) for v in z], name="calibrated")
    narrow = pd.Series([nd.cdf(v * 2) for v in z], name="overconfident")
    return pl.plot_pit(
        pd.DataFrame({"calibrated": good, "overconfident": narrow}),
        "Probability integral transform",
        bins=10,
    )


@card(
    "Relationships",
    "plot_density2d",
    [
        "60,000 points that a scatter would overplot into a blob; here the dense core and the sparse tails are both visible.",
        "Colourbar is labelled in counts although the shading is logarithmic; empty cells are blank.",
    ],
)
def cmp_density():
    rng = np.random.default_rng(59)
    n = 60_000
    x = rng.standard_t(4, n)
    return pl.plot_density2d(
        pd.DataFrame({"x": x, "y": 0.6 * x + rng.standard_t(4, n)}),
        "x",
        "y",
        "Dense scatter",
        bins=70,
    )
