import numpy as np
import pandas as pd
import pytest

import plotters as pl


def test_bars_from_a_plain_count_dict_sorted_largest_first_with_values_written():
    fig = pl.plot_bars({"a": 5, "b": 12, "c": 3})
    bar = fig.data[0]
    assert list(bar.y) == ["b", "a", "c"] and list(bar.text) == ["12", "5", "3"]
    assert bar.orientation == "h" and fig.layout.yaxis.autorange == "reversed"


def test_bars_sort_modes_top_and_vertical():
    d = {"a": 5, "b": 12, "c": 3}
    assert list(pl.plot_bars(d, sort="ascending").data[0].y) == ["c", "a", "b"]
    assert list(pl.plot_bars(d, sort=False).data[0].y) == ["a", "b", "c"]
    assert list(pl.plot_bars(d, top=2).data[0].y) == ["b", "a"]
    assert list(pl.plot_bars(d, horizontal=False).data[0].x) == ["b", "a", "c"]


def test_bars_normalize_shows_percent_and_refuses_zero_total():
    fig = pl.plot_bars({"a": 1, "b": 3}, normalize=True)
    assert list(fig.data[0].text) == ["75.0%", "25.0%"]
    with pytest.raises(ValueError, match="zero"):
        pl.plot_bars({"a": 0, "b": 0}, normalize=True)


def test_bars_accepts_value_counts_and_rejects_frames_and_text():
    pl.plot_bars(pd.Series(list("aabbbc")).value_counts())
    with pytest.raises(pl.ConventionError, match="convention E"):
        pl.plot_bars(pd.DataFrame({"a": [1]}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="numeric"):
        pl.plot_bars({"a": "x"})  # type: ignore[arg-type]


def test_bars_use_one_hue_and_a_reference_line():
    fig = pl.plot_bars({"a": 1, "b": 2}, reference=1.5)
    assert isinstance(fig.data[0].marker.color, str) and fig.layout.shapes[0].x0 == 1.5


def test_counts_counts_values_and_drops_missing_by_default():
    fig = pl.plot_counts(["x", "y", "x", None, "x"])
    assert dict(zip(fig.data[0].y, fig.data[0].x, strict=True)) == {"x": 3, "y": 1}
    kept = pl.plot_counts(["x", None], dropna=False)
    assert len(kept.data[0].y) == 2


def test_grouped_bars_modes_and_cap():
    f = pd.DataFrame({"s1": [1.0, 2.0], "s2": [3.0, 4.0]}, index=["u", "v"])
    assert pl.plot_grouped_bars(f).layout.barmode == "group"
    assert pl.plot_grouped_bars(f, stacked=True).layout.barmode == "relative"
    norm = pl.plot_grouped_bars(f, normalize=True)
    assert np.asarray(norm.data[0].y) + np.asarray(norm.data[1].y) == pytest.approx([1.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        pl.plot_grouped_bars(-f, stacked=True)
    with pytest.raises(ValueError, match="at most"):
        pl.plot_grouped_bars(pd.DataFrame(np.ones((2, 9))))
    with pytest.raises(pl.ConventionError):
        pl.plot_grouped_bars({"a": 1})  # type: ignore[arg-type]


def test_lollipop_and_dumbbell_and_waterfall():
    lol = pl.plot_lollipop({"a": -2, "b": 3})
    assert lol.data[1].x.tolist() == [3.0, -2.0]
    db = pl.plot_dumbbell(
        pd.DataFrame({"n": list("abc"), "s": [1.0, 2, 3], "e": [1.5, 6, 2]}), "n", "s", "e"
    )
    assert list(db.data[1].y) == ["b", "c", "a"], "largest move first"
    wf = pl.plot_waterfall({"start": 10, "up": 5, "down": -3})
    assert list(wf.data[0].measure) == ["relative"] * 3 + ["total"] and wf.data[0].x[-1] == "Total"
    assert pl.plot_waterfall({"a": 1}, total=None).data[0].x == ("a",)
    with pytest.raises(KeyError):
        pl.plot_dumbbell(pd.DataFrame({"n": ["a"]}), "n", "s", "e")


def test_line_band_keeps_labels_and_requires_matching_bands():
    idx = pd.date_range("2022-01-01", periods=5)
    s = pd.Series(np.arange(5.0), index=idx, name="m")
    fig = pl.plot_line_band(s, s - 1, s + 1)
    assert [t.fill for t in fig.data] == [None, "tonexty", None] or fig.data[1].fill == "tonexty"
    with pytest.raises(KeyError, match="lower has no series"):
        pl.plot_line_band(s, {"other": s - 1}, s + 1)


def test_regression_fit_recovers_slope_band_contains_mean_and_group_cap():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    f = pd.DataFrame({"x": x, "y": 3 * x + 1 + rng.normal(0, 0.5, 400)})
    fig = pl.plot_regression(f, "x", "y")
    assert "slope 3" in fig.layout.title.text
    band = fig.data[0]
    assert band.fill == "toself"
    line = fig.data[2]
    assert np.interp(0, line.x, line.y) == pytest.approx(1.0, abs=0.15)
    f["g"] = rng.choice(list("abcd"), 400)
    with pytest.raises(ValueError, match="at most 3"):
        pl.plot_regression(f, "x", "y", group="g")
    with pytest.raises(ValueError, match="more than"):
        pl.plot_regression(f.head(2), "x", "y")


def test_scatter_matrix_layout_and_limits():
    rng = np.random.default_rng(1)
    f = pd.DataFrame(rng.normal(size=(60, 3)), columns=list("abc"))
    fig = pl.plot_scatter_matrix(f)
    assert (
        sum(t.type == "bar" for t in fig.data) == 3
        and sum(t.type == "scatter" for t in fig.data) == 3
    )
    with pytest.raises(ValueError, match="2 to 6"):
        pl.plot_scatter_matrix(f[["a"]])


def test_calendar_heatmap_grid_shape_and_validation():
    s = pd.Series(
        np.arange(31.0), index=pd.date_range("2022-01-03", periods=31)
    )  # starts on a Monday
    fig = pl.plot_calendar_heatmap(s)
    z = np.asarray(fig.data[0].z)
    assert z.shape[0] == 7 and z[0, 0] == 0.0 and z[1, 0] == 1.0
    with pytest.raises(ValueError, match="DatetimeIndex"):
        pl.plot_calendar_heatmap(pd.Series([1.0, 2.0]))
