import numpy as np
import pandas as pd
import pytest

import plotters as pl
from plotters import matrices as m

MIN_GAP = 0.33  # fill-to-ink luminance floor


def _heat(fig):
    return fig.data[0]


def test_straddling_frame_is_diverging_with_symmetric_range():
    frame = pd.DataFrame([[-1.0, 0.2], [3.0, 0.5]])
    heat = _heat(pl.plot_matrix(frame))
    assert (heat.zmin, heat.zmax) == (-3.0, 3.0)


def test_non_straddling_frame_uses_the_sequential_ramp_on_its_own_range():
    frame = pd.DataFrame([[1.0, 2.0], [3.0, 5.0]])
    heat = _heat(pl.plot_matrix(frame))
    assert (heat.zmin, heat.zmax) == (1.0, 5.0)
    assert heat.colorscale[0][1] == pl.get_style().sequential[0]


def test_center_moves_the_symmetry_point():
    frame = pd.DataFrame([[0.0, 1.0], [2.0, 4.0]])
    heat = _heat(pl.plot_matrix(frame, center=1.0))
    assert (heat.zmin, heat.zmax) == (-2.0, 4.0)


def test_mask_diagonal_blanks_it_and_needs_square():
    frame = pd.DataFrame(np.arange(9.0).reshape(3, 3))
    z = np.asarray(_heat(pl.plot_matrix(frame, mask_diagonal=True)).z)
    assert np.isnan(np.diag(z)).all() and not np.isnan(z[0, 1])
    with pytest.raises(ValueError, match="square"):
        pl.plot_matrix(pd.DataFrame(np.ones((2, 3))), mask_diagonal=True)


def test_value_colour_not_rank_distinguishes_close_ranks():
    frame = pd.DataFrame({"a": [0.971, -0.020], "b": [0.5, 0.4]})
    fig = pl.plot_matrix(frame)
    inks = {a.text: a.font.color for a in fig.layout.annotations}
    assert "0.97" in inks and "-0.02" in inks
    # 0.971 and -0.020 are far apart on the value scale, so their fills differ in position
    heat = _heat(fig)
    assert heat.zmax - heat.zmin > 1.5


def test_cell_text_is_formatted_and_annotated_unless_too_big():
    frame = pd.DataFrame([[1.23456, 2.0]])
    assert [a.text for a in pl.plot_matrix(frame, fmt=".1f").layout.annotations] == ["1.2", "2.0"]
    assert not pl.plot_matrix(frame, annotate=False).layout.annotations
    big = pd.DataFrame(np.random.default_rng(0).normal(size=(31, 31)))
    assert not pl.plot_matrix(big).layout.annotations


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_ink_has_enough_luminance_gap_on_every_fill_of_every_ramp(dark):
    with pl.use_style(dark=dark) as st:
        for ramp in (st.sequential, st.diverging):
            for step in range(41):
                fill = m._ramp_color(ramp, step / 40)
                assert m.luminance_gap(fill) >= MIN_GAP, (fill, step)


def test_dark_diverging_midpoint_gets_light_ink_not_a_threshold_rule():
    with pl.use_style(dark=True) as st:
        mid = st.diverging[len(st.diverging) // 2]
        assert m.cell_ink(mid) == m._LIGHT_INK
    with pl.use_style(dark=False) as st:
        mid = st.diverging[len(st.diverging) // 2]
        assert m.cell_ink(mid) == m._DARK_INK


def test_correlation_takes_data_masks_diagonal_and_is_symmetric_about_zero():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    data["b"] += 0.9 * data["a"]
    heat = _heat(pl.plot_correlation(data))
    z = np.asarray(heat.z)
    assert np.isnan(np.diag(z)).all()
    assert heat.zmin == -heat.zmax
    assert z[0, 1] == pytest.approx(data["a"].corr(data["b"]))


def test_correlation_methods_and_fixed_range():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    data = pd.DataFrame({"x": x, "y": np.exp(3 * x), "z": rng.normal(size=300)})
    pearson = np.asarray(_heat(pl.plot_correlation(data)).z)[0, 1]
    spearman = np.asarray(_heat(pl.plot_correlation(data, method="spearman")).z)[0, 1]
    assert spearman == pytest.approx(1.0) and pearson < 0.95
    heat = _heat(pl.plot_correlation(data, fixed_range=True))
    assert (heat.zmin, heat.zmax) == (-1, 1)


def test_correlation_refuses_a_correlation_matrix_and_non_frames():
    data = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 3)), columns=list("abc"))
    with pytest.raises(pl.ConventionError, match="correlation"):
        pl.plot_correlation(data.corr())
    with pytest.raises(pl.ConventionError):
        pl.plot_correlation(data["a"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="two numeric"):
        pl.plot_correlation(data[["a"]])


def test_table_heatmap_shades_each_column_on_its_own_range_and_keeps_values():
    table = pd.DataFrame(
        {"n": [100.0, 100.0], "mean": [0.001, 0.003], "skew": [-1.2, 2.0]}, index=["x", "y"]
    )
    heat = _heat(pl.plot_table_heatmap(table))
    z = np.asarray(heat.z)
    assert z[:, 1].tolist() == [0.0, 1.0] and z[:, 2].tolist() == [0.0, 1.0]
    assert z[:, 0].tolist() == [0.0, 0.0], "a constant column carries no shading"
    texts = [a.text for a in pl.plot_table_heatmap(table).layout.annotations]
    assert "-1.2" in texts and "100" in texts
    with pytest.raises(ValueError):
        pl.plot_table_heatmap(table, normalize="all")  # type: ignore[arg-type]


def test_contours_takes_an_evaluated_grid_and_can_draw_the_diagonal():
    g = np.linspace(0.5, 3, 30)
    frame = pd.DataFrame(np.subtract.outer(g, g) ** 2, index=g, columns=g)
    fig = pl.plot_contours(frame, diagonal=True)
    assert fig.data[0].type == "contour" and fig.data[0].contours.showlabels
    assert fig.data[1].name == "y = x"
    logged = pl.plot_contours(frame, logx=True, logy=True)
    assert logged.layout.xaxis.type == "log" and logged.layout.yaxis.type == "log"


def test_contours_validates_axes():
    with pytest.raises(ValueError, match="numeric"):
        pl.plot_contours(pd.DataFrame([[1.0]], index=["a"], columns=["b"]))
    with pytest.raises(ValueError, match="positive"):
        pl.plot_contours(
            pd.DataFrame(np.ones((2, 2)), index=[-1.0, 1.0], columns=[1.0, 2.0]), logy=True
        )
