"""The small, common tasks: spans, steps, bubbles, error bars, pareto, and combining figures."""

import numpy as np
import pandas as pd
import pytest

import plotters as pl


@pytest.fixture
def daily():
    return pd.Series(np.arange(60.0), index=pd.date_range("2022-01-01", periods=60), name="s")


# ---- shaded spans and steps -------------------------------------------------------------------


def test_spans_are_shaded_bands_below_the_data_with_a_label(daily):
    a, b = daily.index[10], daily.index[20]
    fig = pl.plot_series(daily, "t", spans={"outage": (a, b)})
    rect = next(s for s in fig.layout.shapes if s.type == "rect")
    assert (rect.x0, rect.x1, rect.layer) == (a, b, "below")
    assert (rect.y0, rect.y1, rect.yref) == (0, 1, "paper")
    assert any(n.text == "outage" for n in fig.layout.annotations)


def test_several_spans_and_log_x_label_position():
    s = pd.Series([1.0, 2.0, 3.0], index=[1.0, 10.0, 100.0], name="s")
    fig = pl.plot_series(s, "t", logx=True, spans={"a": (1.0, 10.0), "b": (10.0, 100.0)})
    assert sum(sh.type == "rect" for sh in fig.layout.shapes) == 2
    label = next(n for n in fig.layout.annotations if n.text == "b")
    assert label.x == pytest.approx(1.0), "annotations on a log axis take log10"


def test_step_draws_a_step_function(daily):
    assert pl.plot_series(daily, "t", step=True).data[0].line.shape == "hv"
    assert pl.plot_series(daily, "t").data[0].line.shape == "linear"


# ---- bubble sizes -----------------------------------------------------------------------------


@pytest.fixture
def bubbles():
    return pd.DataFrame(
        {"x": [1.0, 2, 3, 4], "y": [1.0, 2, 3, 4], "pop": [1.0, 4, 16, 64], "g": list("aabb")}
    )


def test_bubble_area_is_proportional_to_the_value(bubbles):
    sizes = np.asarray(pl.plot_scatter(bubbles, "x", "y", size="pop").data[0].marker.size)
    assert sizes[-1] == pytest.approx(30.0)
    assert (sizes[2] / sizes[3]) ** 2 == pytest.approx(16 / 64, rel=0.01), (
        "area, not diameter, tracks the value"
    )


def test_bubble_floor_note_groups_and_refusals(bubbles):
    fig = pl.plot_scatter(bubbles, "x", "y", size="pop", group="g")
    assert [len(t.marker.size) for t in fig.data] == [2, 2]
    assert "area is proportional to pop" in fig.layout.title.text
    assert (
        min(
            pl.plot_scatter(bubbles.assign(pop=[0.0, 0, 0, 100]), "x", "y", size="pop")
            .data[0]
            .marker.size
        )
        >= 4
    )
    with pytest.raises(ValueError, match="negative"):
        pl.plot_scatter(bubbles.assign(pop=-1.0), "x", "y", size="pop")
    with pytest.raises(KeyError, match="not in columns"):
        pl.plot_scatter(bubbles, "x", "y", size="nope")


def test_shaded_scatter_can_also_be_sized(bubbles):
    fig = pl.plot_shaded_scatter(bubbles, "x", "y", "pop", size="pop")
    assert len(np.asarray(fig.data[0].marker.size)) == 4


# ---- error bars -------------------------------------------------------------------------------


def test_bar_error_bars_follow_the_sort_order():
    fig = pl.plot_bars({"a": 5, "b": 12, "c": 3}, error={"a": 1, "b": 2, "c": 0.5})
    bar = fig.data[0]
    assert list(bar.y) == ["b", "a", "c"]
    assert list(bar.error_x.array) == [2.0, 1.0, 0.5] and bar.error_x.symmetric is False
    assert bar.textposition == "inside", (
        "values sit inside so they do not collide with the whiskers"
    )


def test_bar_asymmetric_and_vertical_and_normalised_errors():
    fig = pl.plot_bars(
        {"a": 10, "b": 30}, error=({"a": 1, "b": 3}, {"a": 2, "b": 6}), horizontal=False, sort=False
    )
    assert list(fig.data[0].error_y.arrayminus) == [1.0, 3.0] and list(
        fig.data[0].error_y.array
    ) == [2.0, 6.0]
    norm = pl.plot_bars({"a": 10, "b": 30}, error={"a": 4, "b": 4}, normalize=True, sort=False)
    assert list(norm.data[0].error_x.array) == pytest.approx([0.1, 0.1]), (
        "errors scale with the values"
    )


def test_bar_error_validation():
    with pytest.raises(KeyError, match="no entry"):
        pl.plot_bars({"a": 1, "b": 2}, error={"a": 0.1})
    with pytest.raises(ValueError, match="negative"):
        pl.plot_bars({"a": 1}, error={"a": -0.1})


def test_grouped_bar_error_bars_and_refusals():
    f = pd.DataFrame({"s1": [1.0, 2.0], "s2": [3.0, 4.0]}, index=["u", "v"])
    fig = pl.plot_grouped_bars(f, error=f * 0.1)
    assert list(fig.data[1].error_y.array) == pytest.approx([0.3, 0.4])
    with pytest.raises(ValueError, match="not available"):
        pl.plot_grouped_bars(f, error=f * 0.1, stacked=True)
    with pytest.raises(ValueError, match="same index"):
        pl.plot_grouped_bars(f, error=f.iloc[:1])


# ---- pareto -----------------------------------------------------------------------------------


def test_pareto_ranks_and_accumulates():
    fig = pl.plot_pareto({"a": 5, "b": 50, "c": 30, "d": 15})
    bars, line = fig.data
    assert list(bars.x) == ["b", "c", "d", "a"]
    assert list(bars.y) == pytest.approx([0.5, 0.3, 0.15, 0.05])
    assert line.y[-1] == pytest.approx(1.0) and np.all(np.diff(line.y) > 0)
    assert fig.layout.shapes[0].y0 == 0.8


def test_pareto_folds_the_tail_and_validates():
    fig = pl.plot_pareto({"a": 50, "b": 30, "c": 10, "d": 6, "e": 4}, top=2, threshold=None)
    assert list(fig.data[0].x) == ["a", "b", "Other"] and fig.data[0].y[-1] == pytest.approx(0.2)
    assert not fig.layout.shapes
    with pytest.raises(ValueError, match="non-negative"):
        pl.plot_pareto({"a": -1, "b": 2})
    with pytest.raises(ValueError, match="zero"):
        pl.plot_pareto({"a": 0, "b": 0})


# ---- plot_grid --------------------------------------------------------------------------------


def _panel(name, ylabel):
    s = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2], name=name)
    return pl.plot_series(s, name, ylabel=ylabel, bands={"level": 2.0})


def test_grid_places_every_figure_with_its_own_axes_and_titles():
    grid = pl.plot_grid({"left": _panel("a", "ya"), "right": _panel("b", "yb")})
    assert [t.name for t in grid.data] == ["a", "b"]
    assert grid.layout.yaxis.title.text == "ya" and grid.layout.yaxis2.title.text == "yb"
    assert {a.text for a in grid.layout.annotations} >= {"left", "right", "level"}


def test_grid_remaps_shapes_and_annotations_to_their_panel():
    grid = pl.plot_grid({"left": _panel("a", "y"), "right": _panel("b", "y")})
    assert [s.xref for s in grid.layout.shapes] == ["x domain", "x2 domain"]
    assert [s.yref for s in grid.layout.shapes] == ["y", "y2"]
    level_notes = [a for a in grid.layout.annotations if a.text == "level"]
    assert [a.xref for a in level_notes] == ["x domain", "x2 domain"]


def test_grid_lists_a_repeated_series_once_and_uses_titles_for_a_list():
    one = pd.Series([1.0, 2.0, 3.0], name="same")
    figs = [pl.plot_series(one, "first"), pl.plot_series(one, "second")]
    grid = pl.plot_grid(figs, legend=True)
    assert [t.showlegend is not False for t in grid.data] == [True, False]
    assert {"first", "second"} <= {a.text for a in grid.layout.annotations}


def test_grid_mixes_figure_kinds_and_keeps_category_axes():
    bars = pl.plot_bars({"u": 1, "v": 2})
    grid = pl.plot_grid({"series": _panel("a", "y"), "bars": bars})
    assert [t.type for t in grid.data] == ["scatter", "bar"]
    assert grid.layout.yaxis2.autorange == "reversed" and grid.layout.yaxis2.type == "category"


def test_grid_refuses_menus_non_cartesian_and_nothing():
    menu = pl.plot_series(
        pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]}), views={"x": ["a"], "y": ["b"]}
    )
    with pytest.raises(ValueError, match="menus"):
        pl.plot_grid({"m": menu})
    import plotly.graph_objects as go

    with pytest.raises(ValueError, match="non-Cartesian"):
        pl.plot_grid({"p": go.Figure(go.Pie(values=[1, 2]))})
    with pytest.raises(ValueError, match="at least one"):
        pl.plot_grid({})


def _pairs(n):
    r = np.random.default_rng(0)
    return pd.DataFrame(
        {"x": r.uniform(0.3, 0.8, n), "y": r.normal(size=n), "s": r.uniform(0.5, 2, n),
         "p": [f"pair{k}" for k in range(n)]}
    )  # fmt: skip


def test_labels_are_written_only_when_few_and_always_on_hover():
    for fn, extra in ((pl.plot_shaded_scatter, {"shade": "s"}), (pl.plot_scatter, {})):
        few = fn(_pairs(10), "x", "y", label="p", **extra).data[0]
        many = fn(_pairs(500), "x", "y", label="p", **extra).data[0]
        assert few.mode == "markers+text" and many.mode == "markers"
        assert "customdata" in many.hovertemplate


def test_scatter_hover_is_closest_point():
    assert pl.plot_scatter(_pairs(20), "x", "y").layout.hovermode == "closest"
    assert pl.plot_shaded_scatter(_pairs(20), "x", "y", "s").layout.hovermode == "closest"


def test_reference_outside_the_data_falls_back_to_sequential():
    seq = pl.plot_shaded_scatter(_pairs(50), "x", "y", "s").data[0].marker
    for kw in ({"reference": 0.0}, {"reference": 0.0, "log_shade": True}):
        m = pl.plot_shaded_scatter(_pairs(50), "x", "y", "s", **kw).data[0].marker
        assert m.colorscale == seq.colorscale  # not diverging: all values sit on one side
    div = pl.plot_shaded_scatter(_pairs(50), "x", "y", "s", reference=1.0).data[0].marker
    assert div.colorscale != seq.colorscale


def test_shade_clip_ignores_outliers_and_is_validated():
    frame = _pairs(200)
    frame.loc[0, "s"] = 1000.0
    clipped = pl.plot_shaded_scatter(frame, "x", "y", "s").data[0].marker
    full = pl.plot_shaded_scatter(frame, "x", "y", "s", clip=0).data[0].marker
    assert clipped.cmax < 10 < full.cmax
    with pytest.raises(ValueError, match="clip"):
        pl.plot_shaded_scatter(frame, "x", "y", "s", clip=0.6)


def test_repeated_labels_are_written_once_per_value_and_groups_stay_distinct():
    """Many points per config, label = group: one text per config, 17 configs all told apart."""
    rng = np.random.default_rng(2)
    n_cfg, n_seed = 17, 6
    sweep = pd.DataFrame(
        {
            "config": np.repeat([f"cfg{i}" for i in range(n_cfg)], n_seed),
            "x": np.repeat(rng.uniform(0.2, 0.9, n_cfg), n_seed) + rng.normal(0, 0.01, n_cfg * n_seed),
            "y": np.repeat(rng.uniform(1, 5, n_cfg), n_seed) + rng.normal(0, 0.05, n_cfg * n_seed),
        }
    )  # fmt: skip
    for fig in (
        pl.plot_tradeoff(sweep, "x", "y", label="config", group="config", logx=True, logy=True),
        pl.plot_scatter(sweep, "x", "y", label="config", group="config"),
    ):
        traces = [t for t in fig.data if t.name and t.name.startswith("cfg")]
        assert len(traces) == n_cfg
        written = [sum(1 for s in t.text if s) for t in traces]
        assert max(written) == 1 and sum(written) > 0, "at most one written label per config"
        assert all(set(t.customdata) == {t.name} for t in traces), "every point named on hover"
        assert len({(t.marker.color, t.marker.symbol) for t in traces}) == n_cfg
