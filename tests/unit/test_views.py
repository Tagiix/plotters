import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import plotters as pl


def _fig(name, ylabel):
    return pl.plot_series(pd.Series(np.arange(5.0), name=name), name, ylabel=ylabel)


def test_first_view_opens_and_buttons_toggle_traces():
    fig = pl.plot_views({"A": _fig("a", "ya"), "B": _fig("b", "yb")}, "Merged")
    assert [t.visible for t in fig.data] == [True, False]
    buttons = fig.layout.updatemenus[0].buttons
    assert buttons[1].args[0]["visible"] == [False, True]


def test_axis_titles_switch_with_the_view():
    fig = pl.plot_views({"A": _fig("a", "ya"), "B": _fig("b", "yb")})
    assert fig.layout.yaxis.title.text == "ya"
    assert fig.layout.updatemenus[0].buttons[1].args[1]["yaxis.title.text"] == "yb"


def test_shapes_and_annotations_switch():
    a = _fig("a", "y").add_shape(type="line", x0=1, x1=1, y0=0, y1=1)
    fig = pl.plot_views({"A": a, "B": _fig("b", "y")})
    buttons = fig.layout.updatemenus[0].buttons
    assert len(buttons[0].args[1]["shapes"]) == 1 and buttons[1].args[1]["shapes"] == []


def test_originally_hidden_traces_stay_hidden_in_their_view():
    a = go.Figure(
        [go.Scatter(y=[1, 2], name="x"), go.Scatter(y=[2, 3], name="y", visible="legendonly")]
    )
    fig = pl.plot_views({"A": a, "B": _fig("b", "y")})
    assert [t.visible for t in fig.data][:2] == [True, "legendonly"]


def test_refuses_existing_menus_and_non_cartesian():
    with_menu = pl.plot_series(
        pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]}), views={"x": ["a"], "y": ["b"]}
    )
    with pytest.raises(ValueError, match="menus"):
        pl.plot_views({"A": with_menu})
    with pytest.raises(ValueError, match="non-Cartesian"):
        pl.plot_views({"A": go.Figure(go.Pie(values=[1, 2]))})
    with pytest.raises(ValueError):
        pl.plot_views({})


def test_single_view_has_no_dropdown():
    assert not pl.plot_views({"A": _fig("a", "y")}).layout.updatemenus


def test_annotate_views_changes_with_dropdown_and_keeps_existing():
    fig = pl.plot_views({"A": _fig("a", "y"), "B": _fig("b", "y")})
    pl.annotate_views(fig, {"A": "n=1", "B": "n=2"})
    assert fig.layout.annotations[-1].text == "n=1"
    assert fig.layout.updatemenus[0].buttons[1].args[1]["annotations"][-1]["text"] == "n=2"


def test_annotate_views_on_plot_series_views_and_errors():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]})
    fig = pl.plot_series(df, "t", views={"x": ["a"], "y": ["b"]}, bands={"t": 1.5})
    pl.annotate_views(fig, {"x": "one"})
    btn = fig.layout.updatemenus[0].buttons[1].args[1]["annotations"]
    assert any(a["text"] == "t" for a in btn), "existing annotations are kept"
    with pytest.raises(ValueError, match="no dropdown"):
        pl.annotate_views(pl.plot_series(df, "t"), {"x": "a"})
    with pytest.raises(ValueError, match="unknown views"):
        pl.annotate_views(fig, {"zz": "a"})
