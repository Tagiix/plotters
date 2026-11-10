"""Autocorrelation. Convention A.

The form is deliberately statsmodels' ``plot_acf``: a stem from zero to each coefficient, a marker
at the tip, the significance band as a ribbon behind.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import NormalDist

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import style as _style
from ._types import MAX_SERIES, SeriesLike
from .frames import as_series_dict, pick
from .layout import HEIGHT, finish
from .stats import _finite, acf_values, bartlett_band, pacf_values

__all__ = ["plot_acf", "plot_pacf", "plot_acf_grid"]


def _coefficients(
    values, lags: int, absolute: bool, partial: bool = False
) -> tuple[np.ndarray, int]:
    x = np.abs(np.asarray(values, dtype=float)) if absolute else np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return (pacf_values(x, lags) if partial else acf_values(x, lags)), x.size


def _band(coef: np.ndarray, n: int, alpha: float, partial: bool) -> np.ndarray:
    """Acf: Bartlett's widening band. Pacf: the flat ``z / sqrt(n)`` band (valid under an AR null)."""
    if not partial:
        return bartlett_band(coef, n, alpha)
    z = NormalDist().inv_cdf(1 - alpha / 2)
    return np.full(len(coef), z / np.sqrt(n))


def _stems(lag: np.ndarray, coef: np.ndarray) -> tuple[list[float | None], list[float | None]]:
    xs: list[float | None] = []
    ys: list[float | None] = []
    for k, c in zip(lag, coef, strict=True):
        xs += [float(k), float(k), None]
        ys += [0.0, float(c), None]
    return xs, ys


def _add_panel(
    fig: go.Figure, lag: np.ndarray, coef: np.ndarray, band: np.ndarray | None, color: str,
    name: str, offset: float = 0.0, row: int | None = None, col: int | None = None,
    legend: bool = True,
) -> None:  # fmt: skip
    st = _style.get_style()
    where = {"row": row, "col": col} if row else {}
    if band is not None:
        fig.add_scatter(x=lag, y=band, mode="lines", line={"width": 0}, hoverinfo="skip",
                        showlegend=False, **where)  # fmt: skip
        fig.add_scatter(x=lag, y=-band, mode="lines", line={"width": 0}, fill="tonexty",
                        fillcolor=st.band, hoverinfo="skip", name="significance band",
                        showlegend=legend and name == "", **where)  # fmt: skip
    xs, ys = _stems(lag + offset, coef)
    fig.add_scatter(x=xs, y=ys, mode="lines", line={"color": color, "width": 1.6}, hoverinfo="skip",
                    showlegend=False, legendgroup=name, **where)  # fmt: skip
    fig.add_scatter(x=lag + offset, y=coef, mode="markers", name=name or "acf",
                    marker={"color": color, "size": 6}, showlegend=legend and name != "",
                    legendgroup=name, customdata=lag,
                    hovertemplate="lag %{customdata}: %{y:.3f}<extra></extra>", **where)  # fmt: skip


def plot_acf(
    data: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    lags: int = 40,
    absolute: bool = False,
    alpha: float = 0.05,
    series: Sequence[str] | None = None,
    height: int = HEIGHT,
) -> go.Figure:
    """One correlogram: stems with a marker at the tip and, for one series, the Bartlett band.

    **The band widens with the lag, and that is the point.** Bartlett's formula inherits the
    autocorrelation already present at shorter lags; it equals the textbook ``1.96 / sqrt(n)``
    only at lag 1. The band is pointwise, so about 1 lag in 20 escapes by chance: read the
    profile, not one stem. With several series the stems are placed side by side and no band
    is drawn (it depends on each series' own length and coefficients); use :func:`plot_acf_grid`.

    ``absolute=True`` plots the acf of ``|x|``. On a return series the two ask different
    questions: ``acf(x)`` tests whether the **sign** is predictable, ``acf(|x|)`` whether the
    **magnitude** is, which is volatility clustering. A series can be serially uncorrelated and
    still strongly clustered (the ARCH case).

    Args:
        data: Series, wide DataFrame or ``{label: Series}``.
        title: Figure title.
        subtitle: Muted line under the title.
        lags: Highest lag.
        absolute: Use ``|x|``.
        alpha: Two-sided level of the band.
        series: Subset and order of series.
        height: Figure height in px.

    Raises:
        ValueError: more than ``MAX_SERIES`` series, or ``lags`` too large for the sample.
    """
    return _stem_figure(data, title, subtitle, lags, absolute, alpha, series, height, partial=False)


def plot_pacf(
    data: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    lags: int = 40,
    absolute: bool = False,
    alpha: float = 0.05,
    series: Sequence[str] | None = None,
    height: int = HEIGHT,
) -> go.Figure:
    """The partial autocorrelation: what lag ``k`` adds once the lags in between are accounted for.

    The companion to :func:`plot_acf`, with the same stems and ribbon. For an AR(p) series the pacf
    cuts off after lag p, where the plain acf only decays, so the two together suggest the order.
    The band here is the flat ``z / sqrt(n)`` that holds under an autoregressive null (unlike the
    acf's Bartlett band, it does not widen). It is pointwise: about 1 lag in 20 escapes by chance.
    With several series the stems sit side by side and no band is drawn; use
    ``plot_acf_grid(partial=True)`` for a band per series.

    Args:
        data: Series, wide DataFrame or ``{label: Series}``.
        title: Figure title.
        subtitle: Muted line under the title.
        lags: Highest lag.
        absolute: Use ``|x|``.
        alpha: Two-sided level of the band.
        series: Subset and order of series.
        height: Figure height in px.

    Raises:
        ValueError: more than ``MAX_SERIES`` series, or ``lags`` too large for the sample.
    """
    return _stem_figure(data, title, subtitle, lags, absolute, alpha, series, height, partial=True)


def _stem_figure(
    data: SeriesLike,
    title: str,
    subtitle: str,
    lags: int,
    absolute: bool,
    alpha: float,
    series: Sequence[str] | None,
    height: int,
    *,
    partial: bool,
) -> go.Figure:
    caller = "plot_pacf" if partial else "plot_acf"
    frames = pick(as_series_dict(data, _caller=caller), series)
    if len(frames) > MAX_SERIES:
        raise ValueError(
            f"{caller} draws at most {MAX_SERIES} series; use plot_acf_grid, which has no cap"
        )
    fig = go.Figure()
    lag = np.arange(1, lags + 1, dtype=float)
    single = len(frames) == 1
    for i, (name, s) in enumerate(frames.items()):
        coef, n = _coefficients(_finite(s).to_numpy(), lags, absolute, partial)
        offset = 0.0 if single else (i - (len(frames) - 1) / 2) * 0.7 / len(frames)
        band = _band(coef, n, alpha, partial) if single else None
        _add_panel(fig, lag, coef, band, _style.color(i), "" if single else str(name), offset)
    fig.add_hline(y=0, line={"color": _style.get_style().axis, "width": 1})
    kind = "|x|" if absolute else "x"
    label = "pacf" if partial else "acf"
    return finish(fig, title, subtitle, xlabel="lag", ylabel=f"{label}({kind})", height=height,
                  legend=True if single else None)  # fmt: skip


def plot_acf_grid(
    data: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    lags: int = 40,
    ncols: int = 2,
    absolute: bool = False,
    partial: bool = False,
    alpha: float = 0.05,
    series: Sequence[str] | None = None,
    panel_height: int = 260,
    height: int | None = None,
) -> go.Figure:
    """One correlogram panel per series, **each with its own band**. No series cap.

    A grid rather than an overlay because the band depends on the series' own length and its own
    shorter-lag coefficients, so a shared band would be wrong for all but one series. Colour
    names nothing here (one series per panel), so every panel uses the same hue.

    Args:
        data: Series, wide DataFrame or ``{label: Series}``.
        title: Figure title.
        subtitle: Muted line under the title.
        lags: Highest lag.
        ncols: Panels per row.
        absolute: Use ``|x|`` in every panel.
        partial: Draw partial autocorrelations (flat band) instead of autocorrelations.
        alpha: Two-sided level of the bands.
        series: Subset and order of series.
        panel_height: Height of one panel row in px.
        height: Total figure height in px; overrides the height computed from ``panel_height``.
    """
    frames = pick(as_series_dict(data, _caller="plot_acf_grid"), series)
    names = [str(n) for n in frames]
    rows = -(-len(names) // ncols)
    fig = make_subplots(rows=rows, cols=ncols, subplot_titles=names, horizontal_spacing=0.07,
                        vertical_spacing=min(0.12, 0.5 / rows), shared_yaxes=False)  # fmt: skip
    st = _style.get_style()
    lag = np.arange(1, lags + 1, dtype=float)
    for k, s in enumerate(frames.values()):
        coef, n = _coefficients(_finite(s).to_numpy(), lags, absolute, partial)
        _add_panel(fig, lag, coef, _band(coef, n, alpha, partial), st.accent, "", row=k // ncols + 1,
                   col=k % ncols + 1, legend=False)  # fmt: skip
    fig.update_annotations(font={"size": 11, "color": st.ink_2})
    fig.update_xaxes(title_text="lag", row=rows)
    kind = "|x|" if absolute else "x"
    fig.update_yaxes(title_text=f"{'pacf' if partial else 'acf'}({kind})", col=1)
    return finish(
        fig, title, subtitle, height=height or max(HEIGHT, 120 + panel_height * rows), legend=False
    )
