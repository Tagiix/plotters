"""Gallery cards for the comparison charts (convention B)."""

import numpy as np
import pandas as pd

import plotters as pl

from . import datasets as ds
from .registry import card


def _models() -> pd.DataFrame:
    rng = np.random.default_rng(20)
    rows = []
    for family, base in (("ewrms", 0.9), ("har", 0.7), ("range", 0.8)):
        for k, window in enumerate((10, 20, 60)):
            cost = window * rng.uniform(0.8, 1.2)
            rows.append({"model": f"{family}_{window}", "cost": cost, "error": base / np.sqrt(window) + rng.normal(0, 0.02) + 0.05 * k,
                         "window": window, "ctx": "all"})  # fmt: skip
    return pd.DataFrame(rows)


@card(
    "Comparisons",
    "plot_tradeoff",
    [
        "Dashed Pareto frontier with ringed points; subtitle states which corner wins.",
        "Colour means family (ewrms / har / range), shape means member; each point is labelled.",
    ],
)
def cmp_tradeoff():
    t = _models()
    return pl.plot_tradeoff(t, "cost", "error", "Cost versus error", label="model", group="model",
                            groups={"ewrms": ("ewrms",), "har": ("har",), "range": ("range",)}, logx=True)  # fmt: skip


@card(
    "Comparisons",
    "plot_tradeoff: connect",
    [
        "Each family is one path joined along the window length; the frontier is still drawn across all of them.",
    ],
)
def cmp_tradeoff_connect():
    t = _models()
    return pl.plot_tradeoff(t, "cost", "error", "Paths along the window", group="model",
                            groups={"ewrms": ("ewrms",), "har": ("har",), "range": ("range",)}, connect="window")  # fmt: skip


@card(
    "Comparisons",
    "plot_tradeoff: higher is better",
    [
        "lower_is_better=(True, False): cost low, score high wins; the frontier runs up and to the left.",
    ],
)
def cmp_tradeoff_mixed():
    t = _models().assign(score=lambda d: 1 - d["error"])
    return pl.plot_tradeoff(
        t, "cost", "score", "Cost versus score", label="model", lower_is_better=(True, False)
    )


@card(
    "Comparisons",
    "plot_scatter",
    [
        "Two groups in two hues and shapes; dotted y = x diagonal.",
    ],
)
def cmp_scatter():
    rng = np.random.default_rng(21)
    t = pd.DataFrame({"a": rng.uniform(0, 1, 60), "model": np.tile(["m1", "m2"], 30)})
    t["b"] = t["a"] * np.where(t["model"] == "m1", 1.0, 1.25) + rng.normal(0, 0.06, 60)
    return pl.plot_scatter(t, "a", "b", "Prediction vs. realised", group="model", diagonal=True)


@card(
    "Comparisons",
    "plot_shaded_scatter",
    [
        "Continuous third variable as colour with a colourbar labelled in data units (log-scaled ratio, diverging about 1).",
    ],
)
def cmp_shaded():
    rng = np.random.default_rng(22)
    t = pd.DataFrame({"x": rng.uniform(1, 100, 80), "y": rng.uniform(0.1, 1, 80)})
    t["ratio"] = np.exp(rng.normal(0, 0.8, 80))
    return pl.plot_shaded_scatter(
        t, "x", "y", "ratio", "Colour is a ratio", log_shade=True, reference=1.0, logx=True
    )


@card(
    "Comparisons",
    "plot_rank_heatmap",
    [
        "Rows ordered by mean rank (best at the top); cell shows the rank and the colour follows it; hover shows the value.",
    ],
)
def cmp_rank():
    rng = np.random.default_rng(23)
    frame = pd.DataFrame([{"model": m, "context": c, "error": rng.uniform(0.5, 1.5) * (1 + 0.15 * i)}
                          for i, m in enumerate(["ewrms", "har", "range", "garch", "naive"])
                          for c in ["EUR", "USD", "JPY", "GBP"]])  # fmt: skip
    return pl.plot_rank_heatmap(frame, "model", "context", "error", "Which model is best where?")


@card(
    "Comparisons",
    "plot_box_series",
    [
        "A box at each x with medians joined into a curve per group; boxes of the two groups sit side by side without overlapping.",
        "The x axis is linear and ticked only at the x values that exist (1, 2, 5, 10, 20).",
    ],
)
def cmp_box_series():
    rng = np.random.default_rng(24)
    xs = np.array([1, 2, 5, 10, 20])
    frame = pd.DataFrame([{"window": x, "model": m, "error": rng.lognormal(np.log(1 + 3 / x) + (0.2 if m == "B" else 0), 0.25)}
                          for x in xs for m in ("A", "B") for _ in range(25)])  # fmt: skip
    return pl.plot_box_series(frame, "window", "error", "Error by window", group="model")


@card(
    "Comparisons",
    "plot_small_multiples",
    [
        "One panel per value, shared legend appears once, axis labels only on the outer panels, shared y range.",
    ],
)
def cmp_multiples():
    rng = np.random.default_rng(25)
    frame = pd.DataFrame([{"ticker": t, "k": k, "model": m, "value": (1 + 0.3 * j) / np.sqrt(k) + rng.normal(0, 0.02)}
                          for j, t in enumerate(["AAA", "BBB", "CCC", "DDD"]) for k in range(1, 11) for m in ("m1", "m2")])  # fmt: skip
    return pl.plot_small_multiples(
        frame, "ticker", "k", "value", "Error by horizon", group="model", ncols=2
    )


@card(
    "Comparisons",
    "plot_paired_difference",
    [
        "Cumulative difference against the named baseline: a steady slope, or a single cliff?",
    ],
)
def cmp_paired():
    r = ds.returns(3)
    r["delta"] = r["alpha"] + np.r_[np.zeros(200), np.full(len(r) - 200, 0.004)]
    return pl.plot_paired_difference(r[["alpha", "beta", "delta"]], "alpha", "Versus alpha")


@card(
    "Comparisons",
    "plot_interval",
    [
        "Intervals crossing the dotted reference at 0 are hollow and faded (not separable); the rest are solid.",
        "Rows ordered by the point estimate, best at the top.",
    ],
)
def cmp_interval():
    rng = np.random.default_rng(26)
    pt = rng.normal(0, 0.8, 9)
    frame = pd.DataFrame({"model": [f"model {i}" for i in range(9)], "point": pt, "low": pt - rng.uniform(0.3, 1.0, 9),
                          "high": pt + rng.uniform(0.3, 1.0, 9)})  # fmt: skip
    return pl.plot_interval(
        frame, "low", "high", "point", "Difference to baseline", label="model", reference=0.0
    )
