import math

import plotly.graph_objects as go
import pytest

import plotters as pl


def _fig(n, **trace):
    return go.Figure([go.Scatter(x=[0, 1], y=[0, 1], name=f"s{i}", **trace) for i in range(n)])


def test_legend_below_plot_with_constant_pixel_offset_at_any_height():
    offsets = []
    for height in (300, 420, 900, 1400):
        fig = pl.finish(_fig(3), "t", "sub", height=height)
        margin_b = fig.layout.margin.b
        plot_px = height - fig.layout.margin.t - margin_b
        offsets.append(-fig.layout.legend.y * plot_px)
    assert all(math.isclose(o, offsets[0], rel_tol=1e-6) for o in offsets)


def test_single_series_hides_legend_unless_forced():
    assert pl.finish(_fig(1), "t").layout.showlegend is False
    assert pl.finish(_fig(1), "t", legend=True).layout.showlegend is True
    assert pl.finish(_fig(3), "t").layout.showlegend is True


def test_hidden_traces_are_not_counted_for_legend_space():
    visible = pl.finish(_fig(2), "t")
    many_hidden = go.Figure(
        [go.Scatter(name=f"s{i}", visible=(i < 2) and True or False) for i in range(200)]
    )
    fig = pl.finish(many_hidden, "t")
    assert fig.layout.margin.b == visible.layout.margin.b


def test_many_series_move_legend_to_the_right():
    fig = pl.finish(_fig(30), "t")
    assert fig.layout.legend.orientation == "v"
    assert fig.layout.legend.x > 1


def test_menu_reserves_top_margin():
    plain = pl.finish(_fig(2), "t")
    menu = pl.finish(_fig(2), "t", menu=True)
    assert menu.layout.margin.t > plain.layout.margin.t


def test_title_is_left_aligned_with_muted_subtitle():
    fig = pl.finish(_fig(2), "Title", "sub")
    assert fig.layout.title.x == 0 and fig.layout.title.xanchor == "left"
    assert "sub" in fig.layout.title.text and "<sub>" in fig.layout.title.text


def test_log_ticks_cross_one_with_full_labels():
    vals, text = pl.log_ticks(0.2, 10)
    assert text == ["0.2", "0.5", "1", "2", "5", "10"]
    assert vals[0] == pytest.approx(0.2)


def test_log_ticks_rejects_non_positive():
    with pytest.raises(ValueError):
        pl.log_ticks(0, 10)


def test_apply_log_range_in_log10_ticks_in_data_units():
    fig = pl.apply_log(go.Figure(), "y", 0.2, 10)
    ax = fig.layout.yaxis
    assert ax.type == "log"
    assert list(ax.range) == pytest.approx([math.log10(0.2), 1.0])
    assert list(ax.tickvals) == pytest.approx([0.2, 0.5, 1, 2, 5, 10])
