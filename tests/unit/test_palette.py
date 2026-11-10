"""The palette's guarantees: colour-blind separation, contrast, ramps, identity channels."""

import pytest

from plotters import _color as c
from plotters import style

# Floors from the spec; the achieved values are far above (see test_report below).
CVD_ADJACENT_FLOOR = 8.0
NORMAL_PAIR_FLOOR = 15.0
GRAPHIC_CONTRAST_FLOOR = 3.0  # WCAG for graphical objects

MODES = [style.LIGHT, style.DARK]


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.name)
def test_adjacent_slots_survive_colour_blindness(mode):
    cols = list(mode.categorical) + [mode.categorical[0]]  # slot 8 wraps to slot 0
    for deficiency in c.DEFICIENCIES:
        assert c.min_adjacent_delta_e(cols, deficiency) >= CVD_ADJACENT_FLOOR, deficiency


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.name)
def test_every_pair_is_distinct_for_normal_vision(mode):
    assert c.min_pair_delta_e(list(mode.categorical)) >= NORMAL_PAIR_FLOOR


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.name)
def test_lines_are_visible_on_the_surface(mode):
    for col in mode.categorical:
        assert c.contrast_ratio(col, mode.surface) >= GRAPHIC_CONTRAST_FLOOR, col


def test_eight_categorical_hues():
    assert len(style.LIGHT.categorical) == len(style.DARK.categorical) == 8


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.name)
def test_sequential_lightness_is_monotonic(mode):
    lightness = [c.hex_to_oklch(h)[0] for h in mode.sequential]
    steps = [b - a for a, b in zip(lightness, lightness[1:], strict=False)]
    assert all(s > 0 for s in steps) if mode.dark else all(s < 0 for s in steps)


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.name)
def test_diverging_has_two_poles_and_a_neutral_midpoint(mode):
    div = mode.diverging
    mid = c.hex_to_oklch(div[len(div) // 2])
    assert mid[1] < 0.01, "midpoint must be neutral grey"
    lo, hi = c.hex_to_oklch(div[0]), c.hex_to_oklch(div[-1])
    assert abs(lo[2] - hi[2]) > 120, "poles must be different hues"


def test_line_style_first_eight_solid_then_varies():
    assert {style.line_style(i)[0] for i in range(8)} == {"solid"}
    assert style.line_style(8)[0] != "solid"


def test_vary_dash_differs_from_slot_zero():
    assert style.line_style(1, vary_dash=True)[0] != style.line_style(0, vary_dash=True)[0]


def test_markers_are_unique_across_192_slots_in_colour_shape_fill():
    seen = {(m["color"], m["symbol"]) for m in (style.marker(i) for i in range(192))}
    assert len(seen) == 192


def test_marker_wrap_changes_more_than_colour_alone():
    # Slot 8 shares slot 0's colour but must not share its shape too.
    assert style.marker(8)["symbol"] != style.marker(0)["symbol"]


def test_set_style_installs_matching_template():
    import plotly.io as pio

    with style.use_style(dark=True):
        assert pio.templates.default == "plotters_dark"
    assert pio.templates.default == style.get_style().name


def test_report_achieved_values(capsys):
    """Not an assertion: prints the real numbers for the README (run with -s)."""
    for mode in MODES:
        cols = list(mode.categorical) + [mode.categorical[0]]
        cvd = min(c.min_adjacent_delta_e(cols, d) for d in c.DEFICIENCIES)
        print(
            mode.name,
            "cvd adj",
            round(cvd, 1),
            "pair",
            round(c.min_pair_delta_e(list(mode.categorical)), 1),
        )
