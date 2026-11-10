import numpy as np
import pandas as pd
import pytest

import plotters as pl
from plotters.timeseries import _resolve_visibility


@pytest.fixture
def wide():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=50)
    return pd.DataFrame(rng.normal(0, 0.01, (50, 4)), index=idx, columns=list("abcd"))


def test_accepts_series_wide_frame_and_dict(wide):
    assert len(pl.plot_series(wide["a"], "t").data) == 1
    assert len(pl.plot_series(wide, "t").data) == 4
    assert len(pl.plot_series({k: wide[k] for k in "ab"}, "t").data) == 2


def test_series_subset_fixes_order_and_colour(wide):
    fig = pl.plot_series(wide, "t", series=["c", "a"])
    assert [t.name for t in fig.data] == ["c", "a"]
    assert fig.data[0].line.color == pl.color(0)


def test_unknown_series_is_a_keyerror(wide):
    with pytest.raises(KeyError, match=r"\['zz'\] not in"):
        pl.plot_series(wide, "t", series=["zz"])


def test_wrong_input_is_a_convention_error():
    with pytest.raises(pl.ConventionError):
        pl.plot_series([1, 2, 3], "t")


def test_no_series_limit_but_only_twenty_start_visible():
    idx = pd.RangeIndex(10)
    many = pd.DataFrame({f"s{i}": np.arange(10.0) for i in range(50)}, index=idx)
    fig = pl.plot_series(many, "t")
    assert len(fig.data) == 50
    states = [t.visible for t in fig.data]
    assert states.count(True) == 20 and states.count("legendonly") == 30


def test_drawn_is_respected():
    many = pd.DataFrame({f"s{i}": np.arange(5.0) for i in range(12)})
    fig = pl.plot_series(many, "t", drawn=8)
    assert [t.visible for t in fig.data].count(True) == 8


def test_visibility_without_views():
    assert _resolve_visibility(["a", "b", "c"], None, 2) == {"": [True, True, "legendonly"]}


def test_visibility_common_baseline_is_in_every_view():
    out = _resolve_visibility(["pooled", "a", "b"], {"A": ["a"], "B": ["b"]}, 20)
    assert out == {"A": [True, True, False], "B": [True, False, True]}


def test_visibility_baselines_count_towards_the_initial_total():
    out = _resolve_visibility(["base", "a", "b"], {"A": ["a", "b"]}, 2)
    assert out["A"] == [True, True, "legendonly"]


def test_visibility_views_become_allowlist_past_max_series():
    names = [f"u{i}" for i in range(9)] + ["a"]
    out = _resolve_visibility(names, {"A": ["a"]}, 20)
    assert out["A"] == [False] * 9 + [True]


def test_unknown_view_member_raises():
    with pytest.raises(KeyError, match="views name series"):
        _resolve_visibility(["a"], {"A": ["nope"]}, 20)


def test_views_build_a_dropdown_with_first_view_open(wide):
    fig = pl.plot_series(wide, "t", views={"one": ["a"], "two": ["b"]})
    menu = fig.layout.updatemenus[0]
    assert [b.label for b in menu.buttons] == ["one", "two"]
    assert [t.visible for t in fig.data][:2] == [True, False]


def test_percent_logy_and_refusal_on_non_positive(wide):
    assert pl.plot_series(wide, "t", percent=True).layout.yaxis.tickformat
    with pytest.raises(ValueError, match="strictly positive"):
        pl.plot_series(wide, "t", logy=True)
    assert pl.plot_series(wide + 1, "t", logy=True).layout.yaxis.type == "log"


def test_cap_on_stacked_and_bars_with_line_only():
    many = pd.DataFrame({f"s{i}": np.arange(1.0, 6.0) for i in range(9)})
    with pytest.raises(ValueError, match="plot_series"):
        pl.plot_stacked(many)


def test_cumulative_drawdown_rolling_have_no_cap_and_start_drawn_up_to_drawn():
    many = pd.DataFrame({f"s{i}": np.linspace(0, 0.1, 30) for i in range(25)})
    for fig in (
        pl.plot_cumulative(many),
        pl.plot_drawdown(many),
        pl.plot_rolling(many, 5),
    ):
        flags = [t.visible for t in fig.data if t.showlegend is not False]
        assert flags.count(True) == 20 and flags.count("legendonly") == 5
    assert [t.visible for t in pl.plot_cumulative(many, drawn=3).data].count(True) == 3
    with pytest.raises(ValueError, match="drawn"):
        pl.plot_drawdown(many, drawn=-1)


def test_cumulative_compound_vs_sum():
    r = pd.Series([0.1, 0.1])
    assert pl.plot_cumulative(r, compound=True).data[0].y.tolist() == pytest.approx([1.1, 1.21])
    assert pl.plot_cumulative(r).data[0].y.tolist() == pytest.approx([1.1, 1.2])


def test_drawdown_is_never_positive(wide):
    fig = pl.plot_drawdown(wide)
    assert all((np.asarray(t.y) <= 1e-12).all() for t in fig.data)


def test_stacked_refuses_negative_parts(wide):
    with pytest.raises(ValueError, match="non-negative"):
        pl.plot_stacked(wide)
    assert len(pl.plot_stacked(wide.abs()).data) == 4


def test_rolling_draws_raw_and_statistic(wide):
    fig = pl.plot_rolling(wide["a"], 5, how="std")
    assert len(fig.data) == 2 and fig.data[0].showlegend is False
    with pytest.raises(ValueError):
        pl.plot_rolling(wide, 5, how="nope")  # type: ignore[arg-type]


def test_bars_with_line_requires_series_bars(wide):
    with pytest.raises(pl.ConventionError, match="convention C"):
        pl.plot_bars_with_line(wide, wide["a"])  # type: ignore[arg-type]
    fig = pl.plot_bars_with_line(wide["a"], {"p1": wide["b"], "p2": wide["c"]})
    assert [t.type for t in fig.data] == ["bar", "scatter", "scatter"]


def test_one_y_axis_only(wide):
    for fig in (pl.plot_series(wide, "t"), pl.plot_bars_with_line(wide["a"], wide["b"])):
        assert "yaxis2" not in fig.layout.to_plotly_json()


def test_views_reuse_colours_from_the_start():
    idx = pd.date_range("2020", periods=5)
    data = pd.DataFrame({n: np.arange(5.0) for n in ["pool", "a1", "a2", "b1", "b2"]}, index=idx)
    fig = pl.plot_series(data, views={"A": ["a1", "a2"], "B": ["b1", "b2"]})
    color = {t.name: t.line.color for t in fig.data}
    assert color["a1"] == color["b1"] and color["a2"] == color["b2"]
    assert len({color["pool"], color["a1"], color["a2"]}) == 3


def test_drawn_zero_means_default_not_hidden():
    idx = pd.date_range("2020", periods=5)
    data = pd.DataFrame({"a": np.arange(5.0), "b": np.arange(5.0)}, index=idx)
    assert [t.visible for t in pl.plot_series(data, drawn=0).data] == [True, True]
    with pytest.raises(ValueError, match="drawn"):
        pl.plot_series(data, drawn=-1)


def test_isolated_points_get_a_marker():
    idx = pd.date_range("2020", periods=6)
    data = pd.DataFrame(
        {"full": np.arange(6.0), "one": [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan]}, index=idx
    )
    full, one = pl.plot_series(data).data
    assert full.mode == "lines"
    assert one.mode == "lines+markers" and list(one.marker.size) == [0, 0, 7, 0, 0, 0]
