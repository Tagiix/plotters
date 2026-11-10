"""Distribution charts. All convention A; each series is a bag of values and the index is ignored,
so these take an unindexed Series happily.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import _color
from . import style as _style
from ._types import MAX_SERIES, SeriesLike
from .frames import as_series_dict, pick
from .layout import HEIGHT, apply_log, finish
from .stats import _finite, sample_expectile

__all__ = [
    "plot_null_test",
    "plot_box",
    "plot_histogram",
    "plot_ecdf",
    "plot_qq",
    "plot_cumulative_share",
    "plot_expectile_curve",
]


def _bags(
    data: SeriesLike, series: Sequence[str] | None, caller: str, cap: bool = True
) -> dict[str, np.ndarray]:
    frames = pick(as_series_dict(data, _caller=caller), series)
    if cap and len(frames) > MAX_SERIES:
        raise ValueError(
            f"{caller} draws at most {MAX_SERIES} series (one per colour), got {len(frames)}; "
            "pass series=[...] to choose."
        )
    bags = {str(n): _finite(s).to_numpy() for n, s in frames.items()}
    empty = [n for n, v in bags.items() if v.size == 0]
    if empty:
        raise ValueError(f"series {empty} have no finite values")
    return bags


def plot_null_test(
    null: pd.Series | np.ndarray | Sequence[float],
    observed: float,
    title: str = "",
    subtitle: str = "",
    *,
    alpha: float = 0.05,
    tail: Literal["two-sided", "left", "right"] = "two-sided",
    bins: int = 60,
    xlabel: str = "statistic",
    height: int = HEIGHT,
) -> go.Figure:
    """A simulated distribution with the tested value on it and the rejection region shaded.

    The picture of a hypothesis test. It reads either way round: a null centred on H0 with
    ``observed`` the statistic, or the statistic's own bootstrap with ``observed = 0``. The p
    value is counted from the distribution (Monte Carlo, ``(1 + hits) / (1 + N)``), not read
    off a normal.

    Args:
        null: Simulated values under the null hypothesis.
        observed: The value being tested.
        title: Figure title.
        subtitle: Muted line under the title.
        alpha: Significance level; the rejection region holds ``alpha`` of the mass.
        tail: ``"two-sided"`` (``alpha / 2`` in each tail), ``"left"`` or ``"right"``.
        bins: Histogram bins.
        xlabel: X axis title.
        height: Figure height in px.

    Raises:
        ValueError: ``alpha`` outside (0, 1), unknown ``tail`` or no finite values.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if tail not in ("two-sided", "left", "right"):
        raise ValueError(f"tail must be 'two-sided', 'left' or 'right', got {tail!r}")
    x = np.asarray(null, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("null has no finite values")
    n = x.size
    low_hits, high_hits = (x <= observed).sum(), (x >= observed).sum()
    p_left, p_right = (1 + low_hits) / (1 + n), (1 + high_hits) / (1 + n)
    p = {"left": p_left, "right": p_right, "two-sided": min(1.0, 2 * min(p_left, p_right))}[tail]
    lo_cut = (
        float(np.quantile(x, alpha / 2 if tail == "two-sided" else alpha))
        if tail != "right"
        else -np.inf
    )
    hi_cut = (
        float(np.quantile(x, 1 - alpha / 2 if tail == "two-sided" else 1 - alpha))
        if tail != "left"
        else np.inf
    )

    st = _style.get_style()
    lo, hi = min(x.min(), observed), max(x.max(), observed)
    pad = (hi - lo) * 0.04
    edges = np.linspace(lo - pad, hi + pad, bins + 1)
    counts, _ = np.histogram(x, edges)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    reject = (centers <= lo_cut) | (centers >= hi_cut)
    colors = [_color.with_alpha(st.negative if r else st.accent, 0.8 if r else 0.5) for r in reject]
    fig = go.Figure(
        go.Bar(x=centers, y=counts / (n * width), width=width, marker={"color": colors, "line": {"width": 0}},
               showlegend=False, hovertemplate="%{x:.4g}<extra></extra>")
    )  # fmt: skip
    fig.add_shape(type="line", xref="x", x0=observed, x1=observed, yref="paper", y0=0, y1=1,
                  line={"color": st.ink, "width": 2})  # fmt: skip
    left_half = observed < (lo + hi) / 2
    fig.add_annotation(xref="x", x=observed, yref="paper", y=0.98, yanchor="top", xanchor="left" if left_half else "right",
                       xshift=5 if left_half else -5, showarrow=False, text="observed",
                       font={"size": 10, "color": st.ink})  # fmt: skip
    verdict = "reject H0" if p < alpha else "cannot reject H0"
    fig.add_annotation(
        xref="paper", x=1, xanchor="right", yref="paper", y=1, yanchor="bottom", showarrow=False, align="right",
        text=f"p = {p:.3g} ({tail}), α = {alpha:g}<br>{verdict}", font={"size": 11, "color": st.ink_2, "family": st.mono},
    )  # fmt: skip
    return finish(
        fig, title, subtitle, xlabel=xlabel, ylabel="density", height=height, legend=False
    )


def plot_box(
    data: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    series: Sequence[str] | None = None,
    sort: bool | Literal["ascending", "descending"] = True,
    reference: float | None = None,
    points: bool = False,
    name_points: bool = False,
    logvalue: bool = False,
    horizontal: bool = True,
    xlabel: str = "",
    ylabel: str = "",
    height: int | None = None,
) -> go.Figure:
    """One box per group, ranked by median: the ranking chart.

    One hue throughout: a box summarises many points, so colour names nothing and the category
    axis does the naming. There is deliberately no violin: a density over a handful of points
    invents modes. ``points=True`` is the honest form below a few hundred observations.

    Args:
        data: Series, wide DataFrame or ``{group: Series}``; each series is a group.
        title: Figure title.
        subtitle: Muted line under the title.
        series: Groups to draw **and their order**. Passing it disables the median sort. On
            the default horizontal layout the first name listed appears at the top.
        sort: ``True`` / ``"ascending"`` puts the smallest median first (top); ``"descending"``
            the largest; ``False`` keeps the input order. Ignored when ``series`` is given.
        reference: Draw a line at this value (e.g. 1.0 for a ratio).
        points: Show every observation next to its box.
        name_points: Write the index label next to each outlier (needs an indexed Series).
        logvalue: Logarithmic value axis, for a tail decades past the box (needs data > 0).
        horizontal: Boxes lie horizontally with the value on x.
        xlabel: X axis title.
        ylabel: Y axis title.
        height: Figure height; defaults to a size that grows with the number of groups.
    """
    frames = pick(as_series_dict(data, _caller="plot_box"), series)
    groups = {str(k): v.dropna() for k, v in frames.items()}
    groups = {k: v[np.isfinite(v.to_numpy(dtype=float))] for k, v in groups.items()}
    if any(v.empty for v in groups.values()):
        raise ValueError("a group has no finite values")
    if series is None and sort:
        descending = sort == "descending"
        groups = dict(sorted(groups.items(), key=lambda kv: kv[1].median(), reverse=descending))
    st = _style.get_style()
    fig = go.Figure()
    for name, v in groups.items():
        common = dict(
            name=name, boxpoints="all" if points else "outliers", jitter=0.5, pointpos=0,
            line={"color": st.accent, "width": 1.5}, fillcolor=_color.with_alpha(st.accent, 0.22),
            marker={"color": st.accent, "size": 4, "opacity": 0.5}, showlegend=False,
            text=[str(i) for i in v.index],
        )  # fmt: skip
        values = v.to_numpy(dtype=float)
        fig.add_trace(
            go.Box(x=values, orientation="h", **common)
            if horizontal
            else go.Box(y=values, **common)
        )
    if name_points:
        for name, v in groups.items():
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            far = v[(v < q1 - 1.5 * (q3 - q1)) | (v > q3 + 1.5 * (q3 - q1))]
            if len(far):
                coords = (
                    {"x": far.to_numpy(), "y": [name] * len(far)}
                    if horizontal
                    else {"x": [name] * len(far), "y": far.to_numpy()}
                )
                fig.add_scatter(**coords, mode="text", text=[str(i) for i in far.index], textposition="top center",
                                textfont={"size": 9, "color": st.muted}, showlegend=False, hoverinfo="skip")  # fmt: skip
    value_axis = "x" if horizontal else "y"
    if reference is not None:
        fig.add_shape(
            type="line", **({"x0": reference, "x1": reference, "yref": "paper", "y0": 0, "y1": 1} if horizontal
                            else {"y0": reference, "y1": reference, "xref": "paper", "x0": 0, "x1": 1}),
            line={"color": st.muted, "width": 1, "dash": "dot"},
        )  # fmt: skip
    if horizontal:
        fig.update_yaxes(autorange="reversed", type="category")  # first group at the top
    if logvalue:
        allv = np.concatenate([v.to_numpy(dtype=float) for v in groups.values()])
        if (allv <= 0).any():
            raise ValueError("logvalue needs strictly positive data")
        apply_log(fig, value_axis, float(allv.min()) / 1.2, float(allv.max()) * 1.2)  # type: ignore[arg-type]
    h = height or (max(260, 70 + 44 * len(groups) + 110) if horizontal else HEIGHT)
    return finish(fig, title, subtitle, xlabel=xlabel, ylabel=ylabel, height=h, legend=False)


def plot_histogram(
    data: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    series: Sequence[str] | None = None,
    bins: int = 50,
    density: bool = False,
    normal: bool = False,
    xlabel: str = "",
    height: int = HEIGHT,
) -> go.Figure:
    """Overlaid distributions on shared bins, with an optional normal fit per series.

    Args:
        data: Series, wide DataFrame or ``{label: Series}``.
        title: Figure title.
        subtitle: Muted line under the title.
        series: Subset and order of series.
        bins: Number of bins (shared by all series).
        density: Scale so the area is 1, making different sample sizes comparable.
        normal: Overlay a normal fitted to each series (mean and standard deviation).
        xlabel: X axis title.
        height: Figure height in px.

    Raises:
        ValueError: more than ``MAX_SERIES`` series or an empty series.
    """
    bags = _bags(data, series, "plot_histogram")
    lo = min(v.min() for v in bags.values())
    hi = max(v.max() for v in bags.values())
    edges = np.linspace(lo, hi, bins + 1) if lo < hi else np.linspace(lo - 0.5, hi + 0.5, bins + 1)
    width = edges[1] - edges[0]
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure()
    for i, (name, v) in enumerate(bags.items()):
        counts, _ = np.histogram(v, edges)
        y = counts / (v.size * width) if density else counts
        c = _style.color(i)
        fig.add_bar(
            x=centers,
            y=y,
            width=width,
            name=name,
            marker={"color": _color.with_alpha(c, 0.5), "line": {"width": 0}},
        )
        if normal and v.std() > 0:
            grid = np.linspace(edges[0], edges[-1], 200)
            pdf = np.exp(-((grid - v.mean()) ** 2) / (2 * v.std() ** 2)) / (
                v.std() * math.sqrt(2 * math.pi)
            )
            fig.add_scatter(x=grid, y=pdf if density else pdf * v.size * width, mode="lines",
                            name=f"{name} normal fit", line={"color": c, "width": 1.6, "dash": "dash"})  # fmt: skip
    fig.update_layout(barmode="overlay")
    return finish(fig, title, subtitle, xlabel=xlabel, ylabel="density" if density else "count", height=height,
                  legend=True if len(bags) > 1 or normal else False)  # fmt: skip


def plot_ecdf(
    data: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    series: Sequence[str] | None = None,
    complementary: bool = False,
    logx: bool = False,
    logy: bool = False,
    xlabel: str = "",
    height: int = HEIGHT,
) -> go.Figure:
    """The empirical distribution function: the distribution with no bin-width choice.

    Better for tails. ``complementary=True`` with ``logy=True`` shows the survival function,
    where a fat tail reads as a slow decay.

    Args:
        data: Series, wide DataFrame or ``{label: Series}``.
        title: Figure title.
        subtitle: Muted line under the title.
        series: Subset and order of series.
        complementary: Plot ``P(X > x)`` instead of ``P(X <= x)``.
        logx: Logarithmic x axis (data must be > 0).
        logy: Logarithmic y axis.
        xlabel: X axis title.
        height: Figure height in px.
    """
    bags = _bags(data, series, "plot_ecdf")
    fig = go.Figure()
    for i, (name, v) in enumerate(bags.items()):
        xs = np.sort(v)
        n = xs.size
        y = np.arange(1, n + 1) / n
        if complementary:
            y = (
                1 - np.arange(0, n) / n
            )  # P(X >= x), so the smallest point is 1 and logs stay finite
        fig.add_scatter(
            x=xs,
            y=y,
            mode="lines",
            name=name,
            line={"color": _style.color(i), "width": 2, "shape": "hv"},
        )
    if logx:
        allx = np.concatenate(list(bags.values()))
        if (allx <= 0).any():
            raise ValueError("logx needs strictly positive data")
        apply_log(fig, "x", float(allx.min()) / 1.2, float(allx.max()) * 1.2)
    if logy:
        fig.update_yaxes(type="log")
    ylabel = "P(X ≥ x)" if complementary else "P(X ≤ x)"
    return finish(
        fig, title, subtitle, xlabel=xlabel, ylabel=ylabel, height=height, legend=len(bags) > 1
    )


def plot_qq(
    data: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    series: Sequence[str] | None = None,
    dist: Literal["normal", "t"] = "normal",
    nu: float | None = None,
    standardize: bool | None = None,
    height: int = HEIGHT,
) -> go.Figure:
    """Quantile-quantile plot against the normal, or Student-t fitted by maximum likelihood.

    The reference line is fitted through the quartiles (the IQR), so tail departures stand out.
    ``dist="t"`` is the one to reach for on fat tails: a normal QQ on a fat-tailed series bends
    at both ends and stops there, saying the normal is wrong but not what is right.

    Args:
        data: Series, wide DataFrame or ``{label: Series}``.
        title: Figure title.
        subtitle: Muted line under the title; for ``dist="t"`` the fitted degrees of freedom
            are appended.
        series: Subset and order of series.
        dist: ``"normal"`` or ``"t"``. The t needs scipy (imported lazily).
        nu: Degrees of freedom for ``dist="t"``; ``None`` fits it per series by maximum
            likelihood.
        standardize: Standardise each series to zero mean and unit variance. Defaults to
            ``True`` for ``dist="t"`` (the t quantiles are scaled to unit variance) and
            ``False`` for the normal.
        height: Figure height in px.

    Raises:
        ImportError: ``dist="t"`` and scipy is not installed.
        ValueError: unknown ``dist`` or ``nu <= 2`` (infinite variance, no unit scaling).
    """
    if dist not in ("normal", "t"):
        raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")
    standardize = (dist == "t") if standardize is None else standardize
    bags = _bags(data, series, "plot_qq")
    t_dist = None
    if dist == "t":
        try:
            from scipy import stats as _sps  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "plot_qq(dist='t') needs scipy: pip install 'plotters[stats]'"
            ) from exc
        t_dist = _sps.t
    fig = go.Figure()
    fitted: list[str] = []
    for i, (name, v) in enumerate(bags.items()):
        x = np.sort((v - v.mean()) / v.std() if standardize else v)
        n = x.size
        probs = (np.arange(1, n + 1) - 0.5) / n
        if dist == "normal":
            nd = NormalDist()
            theo = np.array([nd.inv_cdf(p) for p in probs])
            q = nd.inv_cdf
        else:
            assert t_dist is not None
            df = nu if nu is not None else float(t_dist.fit(v)[0])
            if df <= 2:
                origin = "fitted" if nu is None else "given"
                raise ValueError(
                    f"{origin} nu={df:.3g} <= 2: series {name!r} is too heavy-tailed for a "
                    "Student-t reference (no finite variance to scale to). "
                    "Try plot_ecdf(complementary=True, logy=True) to read the tail directly."
                )
            scale = math.sqrt((df - 2) / df)  # unit-variance t quantiles
            theo = t_dist.ppf(probs, df) * scale
            q = lambda p, df=df, scale=scale: float(t_dist.ppf(p, df) * scale)  # noqa: E731
            fitted.append(f"{name}: ν = {df:.3g}")
        c = _style.color(i)
        fig.add_scatter(x=theo, y=x, mode="markers", name=name, marker=_style.marker(i, size=6))
        q25, q75 = float(np.quantile(x, 0.25)), float(np.quantile(x, 0.75))
        slope = (q75 - q25) / (q(0.75) - q(0.25))
        intercept = q25 - slope * q(0.25)
        span = np.array([theo.min(), theo.max()])
        fig.add_scatter(x=span, y=intercept + slope * span, mode="lines", showlegend=False, hoverinfo="skip",
                        line={"color": c, "width": 1.2, "dash": "dot"})  # fmt: skip
    sub = subtitle
    if fitted:
        sub = (subtitle + " · " if subtitle else "") + "; ".join(fitted)
    ylabel = "sample quantiles" + (" (standardised)" if standardize else "")
    xlabel = "normal quantiles" if dist == "normal" else "Student-t quantiles (unit variance)"
    return finish(
        fig, title, sub, xlabel=xlabel, ylabel=ylabel, height=height, legend=len(bags) > 1
    )


def plot_cumulative_share(
    data: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    series: Sequence[str] | None = None,
    height: int = HEIGHT,
) -> go.Figure:
    """How concentrated a total is, against a "paid evenly" diagonal.

    Observations are sorted largest first; the curve is the share of the total held by the top
    ``x`` of observations. The further above the diagonal, the more a few carry the total.

    Args:
        data: Series, wide DataFrame or ``{label: Series}`` of non-negative values.
        title: Figure title.
        subtitle: Muted line under the title.
        series: Subset and order of series.
        height: Figure height in px.

    Raises:
        ValueError: negative values, or a total of zero.
    """
    bags = _bags(data, series, "plot_cumulative_share")
    st = _style.get_style()
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="paid evenly",
                    line={"color": st.muted, "width": 1.2, "dash": "dot"})  # fmt: skip
    for i, (name, v) in enumerate(bags.items()):
        if (v < 0).any():
            raise ValueError(
                f"series {name!r} has negative values; shares of a total need non-negative data"
            )
        total = v.sum()
        if total == 0:
            raise ValueError(f"series {name!r} sums to zero")
        s = np.sort(v)[::-1]
        share = np.concatenate([[0.0], np.cumsum(s) / total])
        frac = np.arange(0, s.size + 1) / s.size
        fig.add_scatter(
            x=frac, y=share, mode="lines", name=name, line={"color": _style.color(i), "width": 2}
        )
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
    return finish(fig, title, subtitle, xlabel="share of observations (largest first)",
                  ylabel="share of total", height=height, legend=True)  # fmt: skip


def plot_expectile_curve(
    data: SeriesLike,
    title: str = "",
    subtitle: str = "",
    *,
    series: Sequence[str] | None = None,
    taus: Sequence[float] | None = None,
    height: int = HEIGHT,
) -> go.Figure:
    """The tau expectile over each series' OWN mean, against tau. The slope is the reading.

    Answers "do a few large observations carry the mean?". Every curve passes through 1 at
    ``tau = 0.5`` and the only thing to read is the slope of the right half. Unlike a quantile,
    an expectile responds to how *large* the exceedances are, not only how many. Normalising by
    each series' own mean is deliberate: the question is concentration, not level, so series on
    different scales belong on one picture.

    Args:
        data: Series, wide DataFrame or ``{label: Series}`` with a positive mean.
        title: Figure title.
        subtitle: Muted line under the title.
        series: Subset and order of series.
        taus: Levels in (0, 1); default 19 points from 0.05 to 0.95 (includes 0.5).
        height: Figure height in px.

    Raises:
        ValueError: a series has a non-positive mean.
    """
    bags = _bags(data, series, "plot_expectile_curve")
    grid = np.linspace(0.05, 0.95, 19) if taus is None else np.asarray(taus, dtype=float)
    fig = go.Figure()
    for i, (name, v) in enumerate(bags.items()):
        mean = v.mean()
        if mean <= 0:
            raise ValueError(
                f"series {name!r} has mean {mean:.3g}; the expectile ratio needs a positive mean"
            )
        y = [sample_expectile(v, float(t)) / mean for t in grid]
        fig.add_scatter(x=grid, y=y, mode="lines+markers", name=name, line={"color": _style.color(i), "width": 2},
                        marker=_style.marker(i, size=5))  # fmt: skip
    return finish(
        fig,
        title,
        subtitle,
        xlabel="τ",
        ylabel="expectile / own mean",
        height=height,
        legend=len(bags) > 1,
    )
