import numpy as np
import pandas as pd
import pytest

import plotters as pl


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _monthly(rng, years=6, trend=0.02):
    idx = pd.date_range("2018-01-01", periods=12 * years, freq="MS")
    t = np.arange(len(idx))
    return pd.Series(
        np.sin(2 * np.pi * t / 12) + trend * t + rng.normal(0, 0.1, len(t)), index=idx, name="s"
    )


# ---- numbers ----------------------------------------------------------------------------------


def test_ccf_lag_sign_says_who_leads(rng):
    x = rng.normal(size=600)
    y = np.roll(x, 3)  # y[t] = x[t - 3]: y follows x by three steps
    lag, c = pl.ccf_values(x, y, 8)
    assert lag[np.argmax(c)] == -3, "a peak at a negative lag means y follows x"
    lag2, c2 = pl.ccf_values(y, x, 8)
    assert lag2[np.argmax(c2)] == 3


def test_ccf_matches_the_plain_correlation_at_lag_zero_and_validates(rng):
    x, y = rng.normal(size=300), rng.normal(size=300)
    lag, c = pl.ccf_values(x, y, 5)
    assert c[list(lag).index(0)] == pytest.approx(np.corrcoef(x, y)[0, 1], abs=1e-12)
    assert len(lag) == 11 and (lag == np.arange(-5, 6)).all()
    with pytest.raises(ValueError, match="lags"):
        pl.ccf_values(x, y, 300)
    with pytest.raises(ValueError, match="constant"):
        pl.ccf_values(x, np.ones(300), 3)
    with pytest.raises(ValueError, match="same length"):
        pl.ccf_values(x, y[:10], 3)


def test_ccf_aligns_series_on_their_index(rng):
    idx = pd.date_range("2022-01-01", periods=200)
    a = pd.Series(rng.normal(size=200), index=idx)
    b = a.shift(2).iloc[10:]  # a shorter, offset series
    lag, c = pl.ccf_values(a, b, 6)
    assert lag[np.argmax(c)] == -2


def test_periodogram_finds_the_cycle_and_detrending_matters(rng):
    t = np.arange(240)
    x = np.sin(2 * np.pi * t / 12) + 0.05 * t + rng.normal(0, 0.3, 240)
    f, p = pl.periodogram(x)
    assert 1 / f[np.argmax(p)] == pytest.approx(12.0)
    f0, p0 = pl.periodogram(x, detrend=False)
    assert np.argmax(p0) == 0, "an undetrended trend swamps the lowest frequency"
    f2, _ = pl.periodogram(x, spacing=1 / 12)
    assert f2[np.argmax(p)] == pytest.approx(1.0), "monthly data in years: one cycle per year"
    with pytest.raises(ValueError, match="at least 8"):
        pl.periodogram([1.0, 2.0, 3.0])


def test_group_summary_normal_interval_and_n(rng):
    df = pd.DataFrame(
        {"g": np.repeat(list("ab"), 400), "v": np.r_[rng.normal(1, 1, 400), rng.normal(5, 2, 400)]}
    )
    out = pl.group_summary(df, "g", "v")
    a = df.loc[df.g == "a", "v"]
    assert out.loc["a", "point"] == pytest.approx(a.mean())
    assert out.loc["a", "high"] - out.loc["a", "point"] == pytest.approx(
        1.959964 * a.std(ddof=1) / np.sqrt(400), rel=1e-4
    )
    assert out["n"].tolist() == [400.0, 400.0] and out.loc["b", "error"] > out.loc["a", "error"]


def test_group_summary_bootstrap_is_reproducible_and_does_the_median(rng):
    df = pd.DataFrame({"g": np.repeat(list("ab"), 300), "v": rng.lognormal(0, 1, 600)})
    one = pl.group_summary(df, "g", "v", stat="median", method="bootstrap", seed=3)
    two = pl.group_summary(df, "g", "v", stat="median", method="bootstrap", seed=3)
    assert one.equals(two)
    assert (one["low"] < one["point"]).all() and (one["point"] < one["high"]).all()
    with pytest.raises(ValueError, match="bootstrap"):
        pl.group_summary(df, "g", "v", stat="median")
    with pytest.raises(ValueError, match="level"):
        pl.group_summary(df, "g", "v", level=1.5)
    with pytest.raises(ValueError, match="stat"):
        pl.group_summary(df, "g", "v", stat="mode")  # type: ignore[arg-type]


def test_group_summary_feeds_plot_interval(rng):
    df = pd.DataFrame({"g": np.repeat(list("abc"), 100), "v": rng.normal(size=300)})
    summary = pl.group_summary(df, "g", "v").rename_axis("g").reset_index()
    fig = pl.plot_interval(summary, "low", "high", "point", label="g", reference=0.0)
    assert len(fig.data) >= 1


# ---- gaps -------------------------------------------------------------------------------------


def test_gaps_share_missing_per_row_label_and_stretch(rng):
    idx = pd.date_range("2022-01-01", periods=100)
    full = pd.Series(1.0, index=idx, name="full")
    holey = full.copy().rename("holey")
    holey.iloc[:50] = np.nan
    fig = pl.plot_gaps(pd.concat([full, holey], axis=1), bins=10)
    z = np.asarray(fig.data[0].z)
    assert (
        z.shape == (2, 10) and (z[0] == 0).all() and (z[1][:5] == 1).all() and (z[1][5:] == 0).all()
    )
    assert list(fig.data[0].y) == ["full (0% missing)", "holey (50% missing)"]


def test_gaps_count_absent_dates_and_infinities_as_missing_and_sort():
    a = pd.Series([1.0, 2.0, 3.0, 4.0], index=[0, 1, 2, 3], name="a")
    b = pd.Series([1.0, np.inf, 3.0], index=[0, 1, 3], name="b")  # missing index 2 and an inf at 1
    fig = pl.plot_gaps({"a": a, "b": b}, bins=4, sort=True)
    assert list(fig.data[0].y)[0].startswith("b (50% missing)")
    with pytest.raises(pl.ConventionError):
        pl.plot_gaps([1, 2, 3])  # type: ignore[arg-type]


# ---- seasonal ---------------------------------------------------------------------------------


def test_seasonal_one_line_per_year_older_lighter_with_an_average(rng):
    fig = pl.plot_seasonal(_monthly(rng, trend=0.0))
    cycles = [t for t in fig.data if t.name != "average"]
    assert [t.name for t in cycles] == [str(y) for y in range(2018, 2024)]
    assert len(cycles[0].x) == 12 and list(fig.layout.xaxis.ticktext)[:3] == ["Jan", "Feb", "Mar"]
    assert fig.data[-1].name == "average"
    # the sine repeats every year, so every year's line peaks in the same month
    assert all(abs(int(np.nanargmax(t.y)) - 2) <= 1 for t in cycles), (
        "every year peaks around March"
    )


def test_seasonal_max_cycles_quarter_weekday_hour_and_validation(rng):
    s = _monthly(rng, years=15)
    assert len([t for t in pl.plot_seasonal(s, max_cycles=4).data if t.name != "average"]) == 4
    assert len(pl.plot_seasonal(s, period="quarter").data[0].x) == 4
    h = pd.Series(
        rng.normal(size=24 * 20), index=pd.date_range("2022-01-01", periods=24 * 20, freq="h")
    )
    assert len(pl.plot_seasonal(h, period="hour").data[0].x) == 24
    assert list(pl.plot_seasonal(h, period="weekday").layout.xaxis.ticktext) == [
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun",
    ]
    with pytest.raises(ValueError, match="DatetimeIndex"):
        pl.plot_seasonal(pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="period"):
        pl.plot_seasonal(s, period="decade")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="agg"):
        pl.plot_seasonal(s, agg="max")  # type: ignore[arg-type]


# ---- lag, ccf, periodogram ---------------------------------------------------------------------


def test_lag_plot_panels_carry_the_correlation(rng):
    x = np.zeros(500)
    for t in range(1, 500):
        x[t] = 0.8 * x[t - 1] + rng.normal()
    fig = pl.plot_lag(pd.Series(x), lags=[1, 5])
    titles = [a.text for a in fig.layout.annotations]
    assert titles[0].startswith("lag 1 (r = 0.7") and titles[1].startswith("lag 5")
    assert sum(t.mode == "markers" for t in fig.data) == 2
    with pytest.raises(ValueError, match="lags"):
        pl.plot_lag(pd.Series(x), lags=0)
    with pytest.raises(pl.ConventionError):
        pl.plot_lag(pd.DataFrame({"a": x}))  # type: ignore[arg-type]


def test_plot_ccf_peak_flat_band_and_direction_text(rng):
    x = pd.Series(rng.normal(size=800), name="rain")
    y = pd.Series(np.roll(x.to_numpy(), 4) + rng.normal(0, 0.2, 800), name="river")
    fig = pl.plot_ccf(x, y, lags=10)
    markers = next(t for t in fig.data if t.mode == "markers")
    assert markers.x[int(np.argmax(markers.y))] == -4
    upper = next(t for t in fig.data if t.mode == "lines" and t.line.width == 0)
    assert np.ptp(upper.y) < 1e-12 and upper.y[0] == pytest.approx(
        1.959964 / np.sqrt(800), rel=1e-4
    )
    assert "river follows rain" in fig.layout.title.text
    with pytest.raises(pl.ConventionError):
        pl.plot_ccf(x.to_numpy(), y)  # type: ignore[arg-type]


def test_plot_periodogram_labels_the_period_and_supports_a_period_axis(rng):
    t = np.arange(360)
    s = np.sin(2 * np.pi * t / 12) + np.sin(2 * np.pi * t / 30) * 0.6 + rng.normal(0, 0.3, 360)
    fig = pl.plot_periodogram(s, peaks=2)
    labels = [a.text for a in fig.layout.annotations]
    assert any(label.startswith("period 12") for label in labels) and any(
        label.startswith("period 30") for label in labels
    )
    by_period = pl.plot_periodogram(s, x="period")
    assert by_period.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"
    with pytest.raises(ValueError, match="x must"):
        pl.plot_periodogram(s, x="hz")  # type: ignore[arg-type]


# ---- bump, table, range buttons ----------------------------------------------------------------


def test_bump_ranks_each_step_and_puts_one_at_the_top():
    frame = pd.DataFrame({"a": [1.0, 3.0, 2.0], "b": [2.0, 1.0, 3.0], "c": [3.0, 2.0, 1.0]})
    fig = pl.plot_bump(frame)
    ys = {t.name: list(t.y) for t in fig.data}
    assert ys == {"a": [1, 3, 2], "b": [2, 1, 3], "c": [3, 2, 1]}
    assert fig.layout.yaxis.autorange == "reversed" and list(fig.layout.yaxis.tickvals) == [1, 2, 3]
    high = pl.plot_bump(frame, lower_is_better=False)
    assert {t.name: list(t.y) for t in high.data}["a"] == [3, 1, 2]
    given = pl.plot_bump(frame, already_ranks=True)
    assert {t.name: list(t.y) for t in given.data}["a"] == [1, 3, 2]


def test_bump_highlight_greys_out_the_rest_and_labels_ties_apart():
    frame = pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 1.0], "c": [3.0, 2.0]})
    fig = pl.plot_bump(frame, highlight=["a"])
    widths = {t.name: t.line.width for t in fig.data}
    assert widths["a"] > widths["b"] == widths["c"]
    ends = sorted(a.y for a in fig.layout.annotations if a.text in ("a", "b"))
    assert ends[1] - ends[0] > 0, "names of items tied at rank 1 are written apart"
    with pytest.raises(ValueError, match="highlight"):
        pl.plot_bump(frame, highlight=["zz"])
    with pytest.raises(pl.ConventionError):
        pl.plot_bump(frame["a"])  # type: ignore[arg-type]


def test_table_formats_aligns_and_marks_missing():
    t = pd.DataFrame(
        {"name": ["a", "b"], "n": [1200.0, np.nan], "share": [0.123, 0.5]}, index=[1, 2]
    )
    fig = pl.plot_table(t, column_formats={"share": ".1%", "n": ",.0f"})
    table = fig.data[0]
    assert list(table.header.values) == ["index", "name", "n", "share"]
    assert list(table.cells.values[2]) == ["1,200", "\u2014"]
    assert list(table.cells.values[3]) == ["12.3%", "50.0%"]
    assert list(table.cells.align) == ["right", "left", "right", "right"]
    assert list(pl.plot_table(t, index=False).data[0].header.values) == ["name", "n", "share"]
    with pytest.raises(KeyError, match="column_formats"):
        pl.plot_table(t, column_formats={"nope": ".1f"})
    with pytest.raises(pl.ConventionError):
        pl.plot_table(t["n"])  # type: ignore[arg-type]


def test_range_buttons_sit_top_right_clear_of_a_dropdown():
    idx = pd.date_range("2022-01-01", periods=400)
    df = pd.DataFrame({"a": np.arange(400.0), "b": np.arange(400.0)}, index=idx)
    plain = pl.plot_series(df, "t", range_buttons=True)
    labels = [b.label for b in plain.layout.xaxis.rangeselector.buttons]
    assert labels == ["1m", "3m", "6m", "YTD", "1y", "all"]
    assert (
        plain.layout.xaxis.rangeselector.x == 1
        and plain.layout.xaxis.rangeselector.xanchor == "right"
    )
    both = pl.plot_series(df, "t", range_buttons=True, views={"x": ["a"], "y": ["b"]})
    assert both.layout.updatemenus[0].x == 0, (
        "the dropdown stays at the left, the buttons at the right"
    )
    assert not pl.plot_series(df, "t").layout.xaxis.rangeselector.buttons
