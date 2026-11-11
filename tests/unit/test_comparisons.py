import numpy as np
import pandas as pd
import pytest

import plotters as pl
from plotters import comparisons as c


@pytest.fixture
def table():
    rng = np.random.default_rng(0)
    models = [f"{f}_{k}" for f in ("ew", "har", "gk") for k in range(3)]
    return pd.DataFrame({"model": models, "cost": rng.uniform(1, 50, 9), "error": rng.uniform(0.1, 1, 9),
                         "ctx": np.tile(list("xyz"), 3)})  # fmt: skip


@pytest.mark.parametrize(
    "call",
    [
        lambda d: pl.plot_scatter(d, "cost", "error"),
        lambda d: pl.plot_tradeoff(d, "cost", "error"),
        lambda d: pl.plot_shaded_scatter(d, "cost", "error", "error"),
        lambda d: pl.plot_rank_heatmap(d, "model", "ctx", "error"),
        lambda d: pl.plot_small_multiples(d, "ctx", "cost", "error"),
        lambda d: pl.plot_box_series(d, "cost", "error"),
        lambda d: pl.plot_interval(d, "cost", "cost", "cost"),
    ],
)
def test_convention_b_plotters_reject_series_and_dicts_with_a_clear_error(call, table):
    with pytest.raises(pl.ConventionError, match="convention B"):
        call(table["cost"])
    with pytest.raises(pl.ConventionError, match="convention B"):
        call({"cost": table["cost"]})
    wide = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]})
    with pytest.raises(KeyError, match="not in columns"):
        call(wide)


def test_pareto_mask_for_each_direction():
    x = np.array([1.0, 2.0, 3.0, 2.5])
    y = np.array([3.0, 2.0, 1.0, 2.5])
    assert c._pareto_mask(x, y, True, True).tolist() == [True, True, True, False]
    assert c._pareto_mask(x, y, False, False).tolist() == [True, False, True, True]  # maximise both
    # lower x better, higher y better: only the single best-on-both candidates survive
    assert c._pareto_mask(np.array([1.0, 2.0]), np.array([1.0, 5.0]), True, False).tolist() == [
        True,
        True,
    ]
    assert c._pareto_mask(np.array([1.0, 2.0]), np.array([5.0, 1.0]), True, False).tolist() == [
        True,
        False,
    ]


def test_tradeoff_frontier_has_only_nondominated_points(table):
    fig = pl.plot_tradeoff(table, "cost", "error")
    front = next(t for t in fig.data if t.name == "Pareto frontier")
    for fx, fy in zip(front.x, front.y, strict=True):
        beaten = (
            (table["cost"] <= fx)
            & (table["error"] <= fy)
            & ((table["cost"] < fx) | (table["error"] < fy))
        ).any()
        assert not beaten
    assert list(front.x) == sorted(front.x)


def test_lower_is_better_accepts_bool_or_pair_and_rejects_junk(table):
    pl.plot_tradeoff(table, "cost", "error", lower_is_better=False)
    pl.plot_tradeoff(table, "cost", "error", lower_is_better=(True, False))
    with pytest.raises(ValueError, match="lower_is_better"):
        pl.plot_tradeoff(table, "cost", "error", lower_is_better=(True,))  # type: ignore[arg-type]
    sub = pl.plot_tradeoff(table, "cost", "error", lower_is_better=(True, False)).layout.title.text
    assert "lower cost" in sub and "higher error" in sub


def test_tradeoff_groups_collapse_by_longest_prefix_colour_is_family_shape_is_member(table):
    fig = pl.plot_tradeoff(table, "cost", "error", group="model",
                           groups={"EW": ("ew",), "HAR": ("har",), "GK": ("gk",)})  # fmt: skip
    pts = [t for t in fig.data if t.mode == "markers" and t.legendgroup]
    by_family = {}
    for t in pts:
        by_family.setdefault(t.legendgroup, []).append(t)
    assert set(by_family) == {"EW", "HAR", "GK"}
    for traces in by_family.values():
        assert len({t.marker.color for t in traces}) == 1, "colour means family"
        assert len({t.marker.symbol for t in traces}) == len(traces), "shape means member"
    assert len({traces[0].marker.color for traces in by_family.values()}) == 3


def test_many_groups_are_drawn_each_with_its_own_colour_and_shape(table):
    tradeoff = pl.plot_tradeoff(table, "cost", "error", group="model")
    scatter = pl.plot_scatter(table, "cost", "error", group="model")
    for fig in (tradeoff, scatter):
        points = [
            t for t in fig.data if t.mode and "markers" in t.mode and t.name not in (None, "")
        ]
        identity = {
            (t.marker.color, t.marker.symbol) for t in points if t.name != "Pareto frontier"
        }
        assert len(identity) >= 4, "each group is told apart by colour and shape"
    with pytest.raises(ValueError, match="needs group"):
        pl.plot_tradeoff(table, "cost", "error", groups={"a": ("a",)})


def test_longest_prefix_wins_in_tradeoff():
    t = pd.DataFrame({"m": ["ewrms", "ewrms_gk"], "x": [1.0, 2.0], "y": [2.0, 1.0]})
    fig = pl.plot_tradeoff(
        t, "x", "y", group="m", groups={"ewrms": ("ewrms",), "range": ("ewrms_gk",)}
    )
    assert {tr.legendgroup for tr in fig.data if tr.mode == "markers" and tr.legendgroup} == {
        "ewrms",
        "range",
    }


def test_connect_draws_one_path_per_group_in_order(table):
    t = table.assign(step=[3, 1, 2] * 3, fam=["a"] * 3 + ["b"] * 3 + ["c"] * 3)
    fig = pl.plot_tradeoff(t, "cost", "error", group="fam", connect="step")
    paths = [tr for tr in fig.data if tr.mode == "lines" and tr.name != "Pareto frontier"]
    assert len(paths) == 3
    first = t[t.fam == "a"].sort_values("step")
    assert list(paths[0].x) == first["cost"].tolist()


def test_scatter_group_vs_label_and_diagonal(table):
    g = pl.plot_scatter(table.assign(f=["a"] * 5 + ["b"] * 4), "cost", "error", group="f")
    assert [t.name for t in g.data] == ["a", "b"]
    lab = pl.plot_scatter(table, "cost", "error", label="model", diagonal=True)
    assert lab.data[0].mode == "markers+text" and lab.data[1].name == "y = x"
    assert lab.layout.xaxis.title.text == "cost"
    with pytest.raises(ValueError, match="strictly positive"):
        pl.plot_scatter(table.assign(cost=-1.0), "cost", "error", logx=True)


def test_shaded_scatter_log_shade_keeps_data_unit_ticks_and_reference_is_symmetric(table):
    t = table.assign(ratio=np.geomspace(0.25, 4, 9))
    fig = pl.plot_shaded_scatter(t, "cost", "error", "ratio", log_shade=True, reference=1.0)
    m = fig.data[0].marker
    assert m.cmin == pytest.approx(-m.cmax)
    assert "1" in m.colorbar.ticktext and "4" in m.colorbar.ticktext
    assert fig.data[0].marker.colorscale[0][1] == pl.get_style().diverging[0]
    with pytest.raises(ValueError, match="strictly positive"):
        pl.plot_shaded_scatter(t.assign(ratio=-1.0), "cost", "error", "ratio", log_shade=True)


def test_rank_heatmap_ranks_within_each_context_and_orders_by_mean_rank():
    f = pd.DataFrame(
        {"m": list("abc") * 2, "ctx": list("xxxyyy"), "v": [1.0, 2.0, 3.0, 3.0, 1.0, 2.0]}
    )
    fig = pl.plot_rank_heatmap(f, "m", "ctx", "v")
    heat = fig.data[0]
    assert list(heat.y) == ["a", "b", "c"] or list(heat.y) == ["b", "a", "c"]
    texts = {(a.y, a.x): a.text for a in fig.layout.annotations}
    assert texts[("a", "x")] == "1" and texts[("a", "y")] == "3" and texts[("b", "y")] == "1"
    flipped = pl.plot_rank_heatmap(f, "m", "ctx", "v", lower_is_better=False)
    assert {(a.y, a.x): a.text for a in flipped.layout.annotations}[("a", "x")] == "3"


def test_rank_heatmap_has_no_series_cap_and_value_in_hover():
    n = 40
    f = pd.DataFrame({"m": [f"m{i}" for i in range(n)] * 3, "ctx": np.repeat(list("xyz"), n),
                      "v": np.random.default_rng(0).normal(size=3 * n)})  # fmt: skip
    fig = pl.plot_rank_heatmap(f, "m", "ctx", "v")
    assert len(fig.data[0].y) == n and fig.data[0].customdata is not None


def test_box_series_requires_numeric_x_and_uses_a_linear_axis():
    f = pd.DataFrame({"x": ["a", "b"] * 5, "y": np.arange(10.0)})
    with pytest.raises(ValueError, match="numeric"):
        pl.plot_box_series(f, "x", "y")
    ok = pd.DataFrame({"x": np.repeat([1.0, 10.0, 100.0], 10), "y": np.arange(30.0)})
    fig = pl.plot_box_series(ok, "x", "y")
    assert fig.layout.xaxis.type == "linear" and list(fig.layout.xaxis.tickvals) == [
        1.0,
        10.0,
        100.0,
    ]
    assert fig.layout.boxmode != "group"


def test_box_series_width_from_tightest_gap_divided_by_group_count_and_median_aligned():
    rng = np.random.default_rng(0)
    f = pd.DataFrame({"x": np.repeat([1.0, 2.0, 10.0], 40), "y": rng.normal(size=120),
                      "g": np.tile(["a", "b"], 60)})  # fmt: skip
    fig = pl.plot_box_series(f, "x", "y", group="g")
    boxes = [t for t in fig.data if t.type == "box"]
    lines = [t for t in fig.data if t.type == "scatter"]
    assert boxes[0].width == pytest.approx(1.0 * 0.8 / 2 * 0.92)
    for box, line in zip(boxes, lines, strict=True):
        assert sorted(set(np.round(box.x, 9))) == sorted(np.round(line.x, 9)), (
            "median line sits on its own boxes"
        )
    wide = f.assign(g=np.tile(list("abcd"), 30))
    assert pl.plot_box_series(wide, "x", "y", group="g").data[0].width < boxes[0].width


def test_small_multiples_panels_shared_legend_and_many_groups():
    rng = np.random.default_rng(1)
    f = pd.DataFrame({"p": np.repeat(list("uvwz"), 10), "x": np.tile(np.arange(10), 4), "y": rng.normal(size=40),
                      "g": np.tile(["a", "b"], 20)})  # fmt: skip
    fig = pl.plot_small_multiples(f, "p", "x", "y", group="g", ncols=2)
    assert len([a for a in fig.layout.annotations]) == 4
    assert sum(1 for t in fig.data if t.showlegend) == 2
    many = pl.plot_small_multiples(f.assign(g=np.tile(list("abcd"), 10)), "p", "x", "y", group="g")
    assert sum(1 for t in many.data if t.showlegend) == 4, "no group cap"


def test_paired_difference_base_names_a_series_and_accumulates():
    idx = pd.date_range("2022-01-01", periods=4)
    df = pd.DataFrame({"a": [1.0, 1, 1, 1], "b": [2.0, 3, 2, 4]}, index=idx)
    fig = pl.plot_paired_difference(df, "a")
    assert fig.data[0].y.tolist() == [1.0, 3.0, 4.0, 7.0] and fig.data[0].name == "b"
    with pytest.raises(KeyError, match="names a series"):
        pl.plot_paired_difference(df, "zz")
    with pytest.raises(ValueError):
        pl.plot_paired_difference(df[["a"]], "a")
    assert pl.plot_paired_difference({"x": df["a"], "y": df["b"]}, "x").data[0].name == "y"


def test_interval_crossing_reference_is_hollow_and_sorted_best_first():
    f = pd.DataFrame(
        {"n": list("abc"), "lo": [-1.0, 0.2, -0.3], "hi": [0.5, 1.0, 0.4], "pt": [-0.2, 0.6, 0.05]}
    )
    fig = pl.plot_interval(f, "lo", "hi", "pt", label="n", reference=0.0)
    hollow = [t for t in fig.data if t.opacity == 0.55]
    solid = [t for t in fig.data if t.opacity == 1.0]
    assert sorted(y for t in hollow for y in t.y) == ["a", "c"] and [
        y for t in solid for y in t.y
    ] == ["b"]
    assert list(fig.layout.yaxis.categoryarray) == ["a", "c", "b"]
    with pytest.raises(ValueError, match="lo > hi"):
        pl.plot_interval(f.assign(lo=5.0), "lo", "hi", "pt")


def test_tradeoff_frontier_switch(table):
    on = pl.plot_tradeoff(table, "cost", "error")
    off = pl.plot_tradeoff(table, "cost", "error", frontier=False)
    assert any(tr.name == "Pareto frontier" for tr in on.data)
    assert (
        not any(tr.name == "Pareto frontier" for tr in off.data)
        and len(off.data) == len(on.data) - 2
    )
    assert "Pareto frontier" not in off.layout.title.text
    assert (
        pl.plot_tradeoff(table, "cost", "error", frontier=True).to_dict() == on.to_dict()
    )  # default


def test_tradeoff_hover_picks_the_nearest_point(table):
    fig = pl.plot_tradeoff(table, "cost", "error")
    assert fig.layout.hovermode == "closest"
    front = next(t for t in fig.data if t.name == "Pareto frontier")
    assert front.hoverinfo == "skip"


def test_hover_unified_only_for_lines_on_a_shared_axis():
    rng = np.random.default_rng(1)
    days = pd.date_range("2020-01-01", periods=30)
    one = pd.Series(rng.normal(size=30), index=days, name="a")
    two = pd.DataFrame({"a": one, "b": one * 2})
    assert pl.plot_series(one).layout.hovermode == "x unified"
    assert pl.plot_series(two).layout.hovermode == "x unified"
    table = pd.DataFrame({"m": list("abcde"), "lo": [1, 2, 3, 4, 5.0], "hi": [2, 3, 4, 5, 6.0],
                          "v": [1.5, 2.5, 3.5, 4.5, 5.5]})  # fmt: skip
    for fig in (
        pl.plot_interval(table, "lo", "hi", "v", label="m"),
        pl.plot_bars({"a": 1, "b": 2}),
        pl.plot_roc(rng.integers(0, 2, 50), pd.Series(rng.uniform(size=50), name="s")),
    ):
        assert fig.layout.hovermode == "closest"
