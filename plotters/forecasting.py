"""Forecast evaluation: the forecast itself, residual diagnostics, and calibration.

Calibration asks whether a forecast is *honest*: when it says 70%, does it happen 70% of the time;
does the realised value fall below the 90% quantile 90% of the time; are the probability integral
transforms uniform.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import _color
from . import style as _style
from ._errors import convention_error
from ._types import SeriesLike
from .correlogram import _add_panel, _band
from .frames import as_series_dict, pick
from .layout import HEIGHT, finish
from .stats import _f, _finite, acf_values, ljung_box

__all__ = [
    "plot_forecast",
    "plot_residuals",
    "plot_calibration",
    "plot_quantile_calibration",
    "plot_pit",
]

_Z95 = NormalDist().inv_cdf(0.975)


def _x(index: pd.Index) -> pd.Index:
    return index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index


def plot_forecast(
    history: pd.Series,
    forecast: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    lower: SeriesLike | None = None,
    upper: SeriesLike | None = None,
    actual: pd.Series | None = None,
    split: object | None = None,
    series: Sequence[str] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    height: int = HEIGHT,
    width: int | None = None,
) -> go.Figure:
    """History, forecast and prediction interval, with the forecast origin marked.

    Args:
        history: The observed series up to the forecast origin.
        forecast: The forecast(s): a Series, or several as a wide DataFrame or ``{name: Series}``
            to compare models over the same horizon.
        title: Figure title.
        subtitle: Muted line under the title.
        lower: Lower edge of the prediction interval, in the same shape as ``forecast``.
        upper: Upper edge. ``lower`` and ``upper`` come as a pair.
        actual: Realised values over the horizon, drawn dotted so you can see how the forecast did.
        split: Where to mark the forecast origin; defaults to the first forecast index.
        series: Subset and order of the forecasts.
        xlabel: X axis title.
        ylabel: Y axis title.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ConventionError: ``history`` is not a single Series.
        KeyError: ``lower`` / ``upper`` lack a forecast's name.
        ValueError: only one of ``lower`` / ``upper``.
    """
    if not isinstance(history, pd.Series):
        raise convention_error("plot_forecast", "C", history, "history must be a single Series.")
    if (lower is None) != (upper is None):
        raise ValueError("lower and upper come as a pair: pass both or neither")
    fc = pick(as_series_dict(forecast, _caller="plot_forecast"), series)
    single = len(fc) == 1 and isinstance(forecast, pd.Series)
    lo = hi = None
    if lower is not None and upper is not None:
        lo = (
            {next(iter(fc)): lower}
            if single and isinstance(lower, pd.Series)
            else as_series_dict(lower)
        )
        hi = (
            {next(iter(fc)): upper}
            if single and isinstance(upper, pd.Series)
            else as_series_dict(upper)
        )
        for name in fc:
            for which, d in (("lower", lo), ("upper", hi)):
                if name not in d:
                    raise KeyError(f"{which} has no series {name!r}; it has {list(d)}")
    st = _style.get_style()
    fig = go.Figure()
    fig.add_scatter(x=_x(history.index), y=history.to_numpy(), mode="lines", name="history",
                    line={"color": st.ink_2, "width": 1.8})  # fmt: skip
    origin = split if split is not None else min(s.index.min() for s in fc.values())
    for i, (name, s) in enumerate(fc.items()):
        color = _style.color(i)
        if lo is not None and hi is not None:
            fig.add_scatter(x=_x(hi[name].index), y=hi[name].to_numpy(), mode="lines", line={"width": 0},
                            hoverinfo="skip", showlegend=False, legendgroup=str(name))  # fmt: skip
            fig.add_scatter(x=_x(lo[name].index), y=lo[name].to_numpy(), mode="lines", line={"width": 0},
                            fill="tonexty", fillcolor=_color.with_alpha(color, 0.2), hoverinfo="skip",
                            showlegend=False, legendgroup=str(name))  # fmt: skip
        fig.add_scatter(x=_x(s.index), y=s.to_numpy(), mode="lines", name=str(name), legendgroup=str(name),
                        line={"color": color, "width": 2.2})  # fmt: skip
    if actual is not None:
        fig.add_scatter(x=_x(actual.index), y=actual.to_numpy(), mode="lines", name="actual",
                        line={"color": st.ink, "width": 1.4, "dash": "dot"})  # fmt: skip
    fig.add_shape(type="line", xref="x", x0=origin, x1=origin, yref="paper", y0=0, y1=1,
                  line={"color": st.muted, "width": 1, "dash": "dot"})  # fmt: skip
    fig.add_annotation(xref="x", x=origin, yref="paper", y=1, yanchor="top", xanchor="left", showarrow=False,
                       text=" forecast", font={"size": 10, "color": st.muted})  # fmt: skip
    return finish(
        fig, title, subtitle, xlabel=xlabel, ylabel=ylabel, height=height, legend=True, width=width
    )


def plot_residuals(
    residuals: pd.Series,
    title: str = "",
    subtitle: str = "",
    *,
    fitted: pd.Series | None = None,
    lags: int = 30,
    bins: int = 40,
    height: int = 640,
    width: int | None = None,
) -> go.Figure:
    """The four panels to look at first when judging a model: time, distribution, QQ and acf.

    Residuals over time (is there structure or a changing spread?), their histogram against a
    normal with the same mean and spread, a normal QQ plot of the standardised residuals, and the
    autocorrelation with its band (is anything left to predict?). The default subtitle gives the
    moments and the Ljung-Box test; the test's p value needs scipy.

    Args:
        residuals: Actual minus predicted, in time order.
        fitted: The fitted values, aligned with ``residuals``. When given, the first panel plots
            residuals **against the fitted value** instead of over time, which is where a changing
            spread or a missed curve shows up.
        title: Figure title.
        subtitle: Muted line under the title; a default summarises the residuals.
        lags: Lags of the acf panel and of the Ljung-Box test.
        bins: Histogram bins.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ConventionError: ``residuals`` is not a single Series.
        ValueError: too few observations for ``lags``.
    """
    if not isinstance(residuals, pd.Series):
        raise convention_error(
            "plot_residuals", "C", residuals, "Pass the residuals as one Series."
        )
    r = _finite(residuals)
    if len(r) <= lags + 1:
        raise ValueError(f"need more than {lags + 1} residuals for lags={lags}, got {len(r)}")
    st = _style.get_style()
    x = r.to_numpy()
    mean, sd = float(x.mean()), float(x.std())
    first = "Against fitted" if fitted is not None else "Over time"
    fig = make_subplots(rows=2, cols=2, subplot_titles=(first, "Distribution", "Normal QQ", "Autocorrelation"),
                        horizontal_spacing=0.09, vertical_spacing=0.16)  # fmt: skip
    if fitted is not None:
        paired = pd.concat([fitted.rename("f"), r.rename("r")], axis=1, join="inner").dropna()
        fig.add_scatter(x=paired["f"], y=paired["r"], mode="markers", showlegend=False,
                        marker={"size": 4, "color": st.accent, "opacity": 0.7}, row=1, col=1)  # fmt: skip
        fig.update_xaxes(title_text="fitted value", row=1, col=1)
    else:
        fig.add_scatter(x=_x(r.index), y=x, mode="lines+markers", showlegend=False, line={"color": st.accent, "width": 1},
                        marker={"size": 3, "color": st.accent}, row=1, col=1)  # fmt: skip
    fig.add_hline(y=0, line={"color": st.axis, "width": 1}, row=1, col=1)  # type: ignore[arg-type]
    edges = np.linspace(x.min(), x.max(), bins + 1)
    counts, _ = np.histogram(x, edges)
    bin_width = edges[1] - edges[0]
    centers = (edges[:-1] + edges[1:]) / 2
    fig.add_bar(x=centers, y=counts / (len(x) * bin_width), width=bin_width, showlegend=False, hoverinfo="skip",
                marker={"color": _color.with_alpha(st.accent, 0.6), "line": {"width": 0}}, row=1, col=2)  # fmt: skip
    grid = np.linspace(edges[0], edges[-1], 160)
    fig.add_scatter(x=grid, y=np.exp(-((grid - mean) ** 2) / (2 * sd**2)) / (sd * math.sqrt(2 * math.pi)), mode="lines",
                    showlegend=False, hoverinfo="skip", line={"color": st.negative, "width": 1.4, "dash": "dash"},
                    row=1, col=2)  # fmt: skip
    nd = NormalDist()
    z = np.sort((x - mean) / sd)
    theo = np.array([nd.inv_cdf((i - 0.5) / len(z)) for i in range(1, len(z) + 1)])
    fig.add_scatter(
        x=theo,
        y=z,
        mode="markers",
        showlegend=False,
        marker={"size": 4, "color": st.accent},
        row=2,
        col=1,
    )
    span = [float(theo.min()), float(theo.max())]
    fig.add_scatter(x=span, y=span, mode="lines", showlegend=False, hoverinfo="skip",
                    line={"color": st.muted, "width": 1, "dash": "dot"}, row=2, col=1)  # fmt: skip
    coef = acf_values(x, lags)
    _add_panel(fig, np.arange(1, lags + 1, dtype=float), coef, _band(coef, len(x), 0.05, False), st.accent, "",
               row=2, col=2, legend=False)  # fmt: skip
    q, p = ljung_box(x, lags)
    skew, kurt = _f(pd.Series(x).skew()), _f(pd.Series(x).kurt())
    default = f"mean {mean:.3g} · std {sd:.3g} · skew {skew:.2f} · excess kurtosis {kurt:.2f} · Ljung-Box Q({lags}) = {q:.1f}"
    if not math.isnan(p):
        default += f", p = {p:.3g}"
    fig.update_annotations(font={"size": 11, "color": st.ink_2})
    fig.update_xaxes(title_text="theoretical normal quantile", row=2, col=1)
    fig.update_xaxes(title_text="lag", row=2, col=2)
    return finish(fig, title, subtitle or default, height=height, legend=False, width=width)


def _wilson(hits: np.ndarray, n: np.ndarray, z: float = _Z95) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion: well behaved near 0 and 1 and for small n."""
    p = hits / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.clip(centre - half, 0.0, 1.0), np.clip(centre + half, 0.0, 1.0)


def plot_calibration(
    probabilities: SeriesLike,
    outcomes: pd.Series | Sequence[int] | np.ndarray,
    title: str = "",
    subtitle: str = "",
    *,
    bins: int = 10,
    strategy: str = "quantile",
    min_count: int = 5,
    height: int = 520,
    width: int | None = None,
) -> go.Figure:
    """Reliability diagram: when a forecast says 70%, how often does it happen?

    Predictions are grouped into bins; each point is the mean predicted probability against the
    observed frequency, with a 95% Wilson interval. Points on the diagonal are calibrated. Above it
    the event happens more often than forecast, below it less often. A curve **flatter** than the
    diagonal (above it at the low end, below it at the high end) means overconfident forecasts:
    their extremes are more extreme than reality.

    Args:
        probabilities: Predicted probabilities in [0, 1]: a Series, or several models as a wide
            DataFrame or ``{name: Series}``.
        outcomes: What happened (0 or 1), in the same order as the probabilities.
        title: Figure title.
        subtitle: Muted line under the title.
        bins: Number of bins.
        strategy: ``"quantile"`` puts the same number of forecasts in every bin (default, robust
            when forecasts cluster); ``"uniform"`` uses equal-width probability bins.
        min_count: Bins with fewer forecasts are not drawn (their interval says nothing).
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ValueError: probabilities outside [0, 1], outcomes that are not 0 / 1, a length mismatch,
            an unknown ``strategy``.
    """
    if strategy not in ("quantile", "uniform"):
        raise ValueError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")
    models = pick(as_series_dict(probabilities, _caller="plot_calibration"))
    y = np.asarray(outcomes, dtype=float)
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("outcomes must be 0 or 1")
    st = _style.get_style()
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line={"color": st.muted, "width": 1, "dash": "dot"},
                    hoverinfo="skip", showlegend=False)  # fmt: skip
    for i, (name, p) in enumerate(models.items()):
        pv = p.to_numpy(dtype=float)
        if len(pv) != len(y):
            raise ValueError(
                f"model {name!r} has {len(pv)} probabilities but there are {len(y)} outcomes"
            )
        if ((pv < 0) | (pv > 1)).any():
            raise ValueError(f"model {name!r} has probabilities outside [0, 1]")
        edges = (
            np.quantile(pv, np.linspace(0, 1, bins + 1))
            if strategy == "quantile"
            else np.linspace(0, 1, bins + 1)
        )
        which = np.clip(np.searchsorted(edges, pv, side="right") - 1, 0, bins - 1)
        rows = []
        for b in range(bins):
            m = which == b
            if m.sum() >= min_count:
                rows.append((pv[m].mean(), y[m].mean(), int(m.sum()), y[m].sum()))
        if not rows:
            raise ValueError(
                f"no bin of model {name!r} has {min_count} forecasts; lower min_count or bins"
            )
        mx, my, n, hits = (np.array(c, dtype=float) for c in zip(*rows, strict=True))
        lo, hi = _wilson(hits, n)
        color = _style.color(i)
        fig.add_scatter(x=mx, y=my, mode="lines+markers", name=str(name), customdata=n,
                        line={"color": color, "width": 1.8}, marker=_style.marker(i, size=8),
                        error_y={"type": "data", "symmetric": False, "array": hi - my, "arrayminus": my - lo,
                                 "color": _color.with_alpha(color, 0.6), "thickness": 1.4, "width": 3},
                        hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<br>n = %{customdata:.0f}<extra></extra>")  # fmt: skip
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(
        range=[0, 1], tickformat=".0%", scaleanchor="x", scaleratio=1, constrain="domain"
    )
    fig.update_xaxes(constrain="domain")
    return finish(fig, title, subtitle or "above the diagonal: happens more often than forecast; flatter than the diagonal: overconfident",
                  xlabel="predicted probability", ylabel="observed frequency", height=height,
                  legend=len(models) > 1, width=width)  # fmt: skip


def plot_quantile_calibration(
    actual: pd.Series,
    quantiles: pd.DataFrame | Mapping[str, pd.DataFrame],
    title: str = "",
    subtitle: str = "",
    *,
    alpha: float = 0.05,
    height: int = 520,
    width: int | None = None,
) -> go.Figure:
    """How often the realised value falls below each forecast quantile, against its nominal level.

    For an honest forecast the 90% quantile is exceeded 10% of the time, so the observed fraction
    below it is 90%. The ribbon is the ``1 - alpha`` range that sampling noise alone allows around
    the diagonal for ``n`` observations; a point outside it is evidence of miscalibration. (It
    assumes independent observations, which overlapping multi-step forecasts are not.)

    Args:
        actual: The realised values.
        quantiles: Forecast quantiles, one column per level (column labels are the levels, such as
            0.05, 0.5, 0.95), rows aligned with ``actual``. A ``{model: DataFrame}`` dict compares
            several models.
        title: Figure title.
        subtitle: Muted line under the title.
        alpha: Two-sided level of the ribbon.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ConventionError: ``actual`` is not a Series.
        ValueError: column labels that are not levels in (0, 1).
    """
    if not isinstance(actual, pd.Series):
        raise convention_error("plot_quantile_calibration", "C", actual, "actual must be a Series.")
    models = dict(quantiles) if isinstance(quantiles, Mapping) else {"model": quantiles}
    st = _style.get_style()
    z = NormalDist().inv_cdf(1 - alpha / 2)
    fig = go.Figure()
    n_ref = len(actual.dropna())
    taus_ref = np.linspace(0, 1, 101)
    half = z * np.sqrt(taus_ref * (1 - taus_ref) / max(n_ref, 1))
    fig.add_scatter(x=np.r_[taus_ref, taus_ref[::-1]], y=np.r_[taus_ref + half, (taus_ref - half)[::-1]], fill="toself",
                    mode="lines", line={"width": 0}, fillcolor=st.band, hoverinfo="skip", showlegend=False)  # fmt: skip
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False, hoverinfo="skip",
                    line={"color": st.muted, "width": 1, "dash": "dot"})  # fmt: skip
    for i, (name, q) in enumerate(models.items()):
        try:
            levels = np.array([float(c) for c in q.columns])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "quantile columns must be numeric levels such as 0.05, 0.5, 0.95"
            ) from exc
        if ((levels <= 0) | (levels >= 1)).any():
            raise ValueError("quantile levels must lie strictly between 0 and 1")
        order = np.argsort(levels)
        aligned = pd.concat(
            [actual.rename("actual"), q.iloc[:, order]], axis=1, join="inner"
        ).dropna()
        observed = np.array(
            [(aligned["actual"] <= aligned.iloc[:, 1 + k]).mean() for k in range(len(levels))]
        )
        fig.add_scatter(x=levels[order], y=observed, mode="lines+markers", name=str(name),
                        line={"color": _style.color(i), "width": 1.8}, marker=_style.marker(i, size=8))  # fmt: skip
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(
        range=[0, 1], tickformat=".0%", scaleanchor="x", scaleratio=1, constrain="domain"
    )
    fig.update_xaxes(constrain="domain")
    return finish(fig, title, subtitle or f"points outside the ribbon: miscalibrated at the {1 - alpha:.0%} level",
                  xlabel="nominal quantile level", ylabel="observed fraction below", height=height,
                  legend=len(models) > 1, width=width)  # fmt: skip


def plot_pit(
    pit: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    bins: int = 10,
    alpha: float = 0.05,
    height: int = HEIGHT,
    width: int | None = None,
) -> go.Figure:
    """Probability integral transform histogram: is the forecast distribution right?

    If each observation is passed through the forecast's own CDF, an honest forecast gives values
    that are uniform on [0, 1], so the bars sit at density 1. The ribbon is the range sampling noise
    allows. Reading it: a **U** shape means the forecast is too narrow (overconfident), a **hump**
    too wide, and a **slope** biased.

    Args:
        pit: PIT values in [0, 1]: a Series, or several models as a wide DataFrame or dict.
        title: Figure title.
        subtitle: Muted line under the title.
        bins: Number of bins.
        alpha: Two-sided level of the ribbon.
        height: Figure height in px.
        width: Figure width in px; ``None`` fills the container.

    Raises:
        ValueError: values outside [0, 1].
    """
    models = pick(as_series_dict(pit, _caller="plot_pit"))
    st = _style.get_style()
    z = NormalDist().inv_cdf(1 - alpha / 2)
    edges = np.linspace(0, 1, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure()
    n_max = max(len(_finite(s)) for s in models.values())
    spread = z * math.sqrt((1 / bins) * (1 - 1 / bins) / max(n_max, 1)) * bins
    fig.add_scatter(x=[0, 1, 1, 0], y=[1 - spread, 1 - spread, 1 + spread, 1 + spread], fill="toself", mode="lines",
                    line={"width": 0}, fillcolor=st.band, hoverinfo="skip", showlegend=False)  # fmt: skip
    fig.add_hline(y=1, line={"color": st.muted, "width": 1, "dash": "dot"})
    for i, (name, s) in enumerate(models.items()):
        v = _finite(s).to_numpy()
        if ((v < 0) | (v > 1)).any():
            raise ValueError(f"model {name!r} has PIT values outside [0, 1]")
        density = np.histogram(v, edges)[0] / (len(v) * (1 / bins))
        if len(models) == 1:
            fig.add_bar(x=centers, y=density, width=1 / bins, showlegend=False,
                        marker={"color": _color.with_alpha(st.accent, 0.75), "line": {"width": 0}},
                        hovertemplate="%{x:.2f}: %{y:.2f}<extra></extra>")  # fmt: skip
        else:
            fig.add_scatter(x=centers, y=density, mode="lines+markers", name=str(name), line={"color": _style.color(i), "width": 1.8},
                            marker=_style.marker(i, size=7), line_shape="linear")  # fmt: skip
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(rangemode="tozero")
    return finish(fig, title, subtitle or "uniform is calibrated; U = overconfident, hump = too wide, slope = biased",
                  xlabel="PIT value", ylabel="density", height=height, legend=len(models) > 1, width=width)  # fmt: skip
