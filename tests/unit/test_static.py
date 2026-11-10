import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.image import imread

import plotters as pl
from plotters import static as ps
from plotters.matrices import cell_ink


@pytest.fixture(autouse=True)
def _print_style():
    import matplotlib as mpl

    with mpl.rc_context():
        ps.use_print_style()
        yield
    plt.close("all")


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(60, 4)).cumsum(0), index=pd.date_range("2022-01-01", periods=60),
                        columns=["alpha", "beta", "gamma", "delta"])  # fmt: skip


def test_figure_size_is_a_fraction_of_a4_text_width():
    w, h = ps.figure_size(0.86)
    assert w == pytest.approx(0.86 * (21.0 - 4.4) / 2.54)
    assert h == pytest.approx(w * 0.62)
    assert ps.figure_size(1.0, aspect=0.5)[1] == pytest.approx(ps.figure_size(1.0)[0] * 0.5)


def test_save_writes_300_dpi_and_closes_the_figure(tmp_path, frame):
    fig = ps.plot_lines(frame, "t")
    w, h = fig.get_size_inches()
    out = tmp_path / "thing.png"
    ps.save(fig, out)
    img = imread(out)
    assert img.shape[1] == pytest.approx(w * 300, abs=2) and img.shape[0] == pytest.approx(
        h * 300, abs=2
    )
    assert not plt.fignum_exists(fig.number)


def test_print_style_uses_serif_nine_point_and_the_palette():
    import matplotlib as mpl

    assert mpl.rcParams["font.family"] == ["serif"] and mpl.rcParams["font.size"] == 9
    assert mpl.rcParams["axes.prop_cycle"].by_key()["color"] == list(pl.get_style().categorical)


def test_vary_dash_is_on_by_default_for_print(frame):
    ax = ps.plot_lines(frame).axes[0]
    styles = {
        tuple(
            line.get_linestyle()
            if isinstance(line.get_linestyle(), tuple)
            else (line.get_linestyle(),)
        )
        for line in ax.lines
    }
    assert len(styles) > 1, "four solid lines would be one line when photocopied"
    flat = ps.plot_lines(frame, vary_dash=False).axes[0]
    assert {line.get_linestyle() for line in flat.lines} == {"-"}


def test_legend_by_default_and_end_labels_instead_when_asked(frame):
    assert ps.plot_lines(frame).axes[0].get_legend() is not None
    ax = ps.plot_lines(frame, label_last=True).axes[0]
    assert ax.get_legend() is None
    assert sorted(t.get_text() for t in ax.texts if t.get_text() in frame.columns) == sorted(
        frame.columns
    )


def test_end_labels_are_nudged_apart_when_curves_end_at_the_same_height():
    idx = pd.date_range("2022-01-01", periods=30)
    same = pd.DataFrame(
        {n: np.linspace(0, 1, 30) for n in ("one", "two", "three", "four")}, index=idx
    )
    fig = ps.plot_lines(same, label_last=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [
        t.get_window_extent(renderer) for t in fig.axes[0].texts if t.get_text() in same.columns
    ]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert not a.overlaps(b), "two names overprinted"


def test_percent_writes_a_bare_percent_sign(frame):
    ax = ps.plot_lines(frame / 100, percent=True).axes[0]
    ax.figure.canvas.draw()
    labels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert labels and all("%" in label and "\\" not in label for label in labels)


def test_long_subtitle_wraps_to_the_axes_width(frame):
    fig = ps.plot_lines(frame, "title", "an overlong subtitle that would run off the page " * 6)
    fig.canvas.draw()
    ax = fig.axes[0]
    renderer = fig.canvas.get_renderer()
    sub = next(t for t in ax.texts if "overlong" in t.get_text())
    assert "\n" in sub.get_text()
    assert sub.get_window_extent(renderer).width <= ax.get_window_extent(renderer).width * 1.02


def test_points_frontier_group_and_labels():
    t = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 2.5], "y": [3.0, 2.0, 1.0, 2.6], "m": list("abcd"), "g": list("uvuv")}
    )
    ax = ps.plot_points(t, "x", "y", label="m", group="g", frontier=True).axes[0]
    front = next(line for line in ax.lines if line.get_label() == "Pareto front")
    assert list(front.get_xdata()) == [1.0, 2.0, 3.0]
    assert len(ax.collections) == 2 and len(ax.texts) == 4


def test_ranked_bars_are_sorted_and_written():
    ax = ps.plot_ranked_bars({"a": 1.0, "b": 3.0, "c": 2.0}, fmt=".1f").axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["b", "c", "a"]
    assert [t.get_text() for t in ax.texts] == ["3.0", "2.0", "1.0"]


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_heatmap_ink_follows_the_measured_fill(dark):
    from plotters.matrices import _ramp_color

    with pl.use_style(dark=dark) as st:
        ps.use_print_style()
        frame = pd.DataFrame(np.linspace(-1, 1, 12).reshape(3, 4))
        ax = ps.plot_heatmap(frame).axes[0]
        values = frame.to_numpy()
        for text in ax.texts:
            c, r = int(text.get_position()[0]), int(text.get_position()[1])
            fill = _ramp_color(st.diverging, (values[r, c] + 1) / 2)
            assert text.get_color() == cell_ink(fill)


def test_function_marks_and_stems_band_and_panels(frame):
    ax = ps.plot_function(
        {"sq": lambda v: v**2, "abs": np.abs}, np.linspace(-2, 2, 50), marks=[0.0]
    ).axes[0]
    assert len(ax.lines) == 3 and ax.get_legend() is not None
    coef = np.array([0.4, 0.2, -0.1, 0.05])
    stems = ps.plot_stems(coef, band=np.array([0.1, 0.12, 0.14, 0.16])).axes[0]
    assert len(stems.collections) == 2  # the band and the stems
    fig = ps.plot_panels({"A": frame[["alpha", "beta"]], "B": frame[["alpha", "beta"]], "C": frame[["alpha", "beta"]]},
                         ncols=2, xlabel="date", ylabel="level")  # fmt: skip
    assert len(fig.legends) == 1
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3
    assert (
        sum(1 for ax in visible if ax.get_xlabel()) == 2
        and sum(1 for ax in visible if ax.get_ylabel()) == 2
    )
