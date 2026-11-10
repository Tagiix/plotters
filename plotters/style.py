"""Theme: palette, ramps, identity channels (colour / dash / marker) and the plotly template.

Call :func:`set_style` once at the top of a script or notebook. Every figure created
afterwards, plotters' or plain ``go.Figure``, picks up the fonts, colours and grid.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import plotly.graph_objects as go
import plotly.io as pio

from . import _color

__all__ = [
    "Style",
    "LIGHT",
    "DARK",
    "set_style",
    "use_style",
    "get_style",
    "color",
    "colour",
    "line_style",
    "marker",
    "palette",
    "sequential_scale",
    "diverging_scale",
    "template_name",
]

FONT = "Inter, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace"

# The order is the safety mechanism: neighbours were chosen to stay distinguishable under
# protan / deutan / tritan simulation. Re-order or edit a value and
# tests/unit/test_palette.py will tell you whether the guarantees still hold.
_CATEGORICAL_LIGHT = (
    "#2b5cda",  # blue
    "#b03e00",  # rust
    "#109dbe",  # cyan
    "#be1357",  # crimson
    "#c1850b",  # amber
    "#ac81bf",  # lavender
    "#11a57a",  # green
    "#627201",  # olive
)
_CATEGORICAL_DARK = (
    "#4c7ef3",
    "#dd5412",
    "#10bfe7",
    "#db4975",
    "#e9a107",
    "#caa2dc",
    "#16c995",
    "#7e910c",
)

_SEQUENTIAL_HUE = 255.0  # blue
_DIVERGING_HUES = (255.0, 50.0)  # blue ↔ orange; the first pole is the "low" side


def _sequential(dark: bool, steps: int = 9) -> tuple[str, ...]:
    """One hue, monotonic lightness. Light mode: pale → deep. Dark mode: deep → bright."""
    lo, hi = (0.24, 0.86) if dark else (0.97, 0.36)
    out = []
    for i in range(steps):
        t = i / (steps - 1)
        chroma = 0.14 * (1 - abs(2 * t - 1) ** 2.2) + 0.02
        out.append(_color.oklch_to_hex(lo + (hi - lo) * t, chroma, _SEQUENTIAL_HUE))
    return tuple(out)


def _diverging(dark: bool, steps: int = 11) -> tuple[str, ...]:
    """Two poles around a neutral grey midpoint (dark in dark mode, light in light mode)."""
    mid = steps // 2
    pole_l = 0.72 if dark else 0.46
    mid_l = 0.27 if dark else 0.95
    out = []
    for i in range(steps):
        t = abs(i - mid) / mid  # 0 at the midpoint, 1 at a pole
        hue = _DIVERGING_HUES[0] if i < mid else _DIVERGING_HUES[1]
        lightness = mid_l + (pole_l - mid_l) * t
        chroma = 0.17 * t**0.9 if t else 0.0
        out.append(_color.oklch_to_hex(lightness, chroma, hue))
    return tuple(out)


@dataclass(frozen=True)
class Style:
    """Every colour and font token the library draws with, for one mode."""

    name: str
    dark: bool
    surface: str
    ink: str  # titles
    ink_2: str  # body text, axis titles
    muted: str  # ticks, subtitles, footnotes
    grid: str
    axis: str
    categorical: tuple[str, ...]
    sequential: tuple[str, ...]
    diverging: tuple[str, ...]
    positive: str
    negative: str
    band: str  # translucent fill for confidence / significance ribbons
    font: str = FONT
    mono: str = MONO

    @property
    def accent(self) -> str:
        """The first categorical colour, used when one colour is enough."""
        return self.categorical[0]


LIGHT = Style(
    name="plotters",
    dark=False,
    surface="#fbfbfa",
    ink="#14171c",
    ink_2="#3b424d",
    muted="#6b7280",
    grid="#e7e9ec",
    axis="#c8ccd2",
    categorical=_CATEGORICAL_LIGHT,
    sequential=_sequential(False),
    diverging=_diverging(False),
    positive="#11a57a",
    negative="#be1357",
    band="rgba(43, 92, 218, 0.14)",
)

DARK = Style(
    name="plotters_dark",
    dark=True,
    surface="#111318",
    ink="#eceef2",
    ink_2="#c3c8d1",
    muted="#8b93a1",
    grid="#232830",
    axis="#363c47",
    categorical=_CATEGORICAL_DARK,
    sequential=_sequential(True),
    diverging=_diverging(True),
    positive="#16c995",
    negative="#db4975",
    band="rgba(76, 126, 243, 0.20)",
)

_current: Style = LIGHT


def _build_template(style: Style) -> go.layout.Template:
    axis = {
        "gridcolor": style.grid,
        "gridwidth": 1,
        "linecolor": style.axis,
        "zeroline": False,
        "tickfont": {"color": style.muted, "size": 10},
        "title": {"font": {"color": style.ink_2, "size": 11}},
    }
    return go.layout.Template(
        layout={
            "colorway": list(style.categorical),
            "font": {"family": style.font, "size": 11, "color": style.ink_2},
            "paper_bgcolor": style.surface,
            "plot_bgcolor": style.surface,
            "title": {
                "x": 0,
                "xref": "paper",
                "xanchor": "left",
                "font": {"size": 15, "color": style.ink},
            },
            "xaxis": axis,
            "yaxis": axis,
            "hovermode": "x unified",
            "hoverlabel": {"font": {"family": style.font}},
            "colorscale": {
                "sequential": [
                    [i / (len(style.sequential) - 1), c] for i, c in enumerate(style.sequential)
                ],
                "diverging": [
                    [i / (len(style.diverging) - 1), c] for i, c in enumerate(style.diverging)
                ],
            },
        }
    )


for _style in (LIGHT, DARK):
    pio.templates[_style.name] = _build_template(_style)


def template_name() -> str:
    """Name of the plotly template matching the current style."""
    return _current.name


def get_style() -> Style:
    """Return the style currently in force."""
    return _current


def _resolve(dark: bool | Literal["auto"]) -> Style:
    if dark == "auto":
        return DARK if os.environ.get("PLOTTERS_STYLE", "").lower() == "dark" else LIGHT
    return DARK if dark else LIGHT


def set_style(dark: bool | Literal["auto"] = False) -> Style:
    """Choose the light or dark theme for everything drawn from now on.

    Also installs the matching plotly template as the default, so plain ``go.Figure``
    charts (not only plotters') get the same fonts, colours and grid.

    Args:
        dark: ``True`` for the dark theme, ``False`` for light. ``"auto"`` reads the
            ``PLOTTERS_STYLE`` environment variable (``"dark"``) and otherwise picks light.

    Returns:
        The :class:`Style` now in force.

    Example:
        >>> import plotters as pl
        >>> pl.set_style(dark=True)  # once, at the top
    """
    global _current
    _current = _resolve(dark)
    pio.templates.default = _current.name
    return _current


@contextmanager
def use_style(dark: bool | Literal["auto"] = False) -> Iterator[Style]:
    """Temporarily switch theme, restoring the previous one on exit.

    Args:
        dark: Same meaning as in :func:`set_style`.

    Example:
        >>> with pl.use_style(dark=True):
        ...     fig = pl.plot_series(returns, "Daily returns")
    """
    previous = _current
    try:
        yield set_style(dark)
    finally:
        set_style(previous.dark)


def palette() -> tuple[str, ...]:
    """The eight categorical colours of the current style, in slot order."""
    return _current.categorical


def color(i: int) -> str:
    """Categorical colour for series slot ``i``; slots wrap after eight.

    Args:
        i: Zero-based series index.
    """
    cats = _current.categorical
    return cats[i % len(cats)]


colour = color  # British spelling alias


_DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")


def line_style(i: int, *, vary_dash: bool = False) -> tuple[str, float]:
    """Dash pattern and width for series slot ``i``.

    The first eight slots are solid, one per colour (on screen the legend isolates a
    series on click). Past eight the dash changes on each wrap so a repeated colour is
    still a different line. Set ``vary_dash`` for print, where there is no legend to click
    and colour alone is not enough.

    Args:
        i: Zero-based series index.
        vary_dash: Cycle the dash from slot 0 instead of only after the colours wrap.

    Returns:
        ``(plotly dash name, line width in px)``.
    """
    n = len(_current.categorical)
    dash = _DASHES[i % len(_DASHES)] if vary_dash else _DASHES[(i // n) % len(_DASHES)]
    return dash, 2.0


_SHAPES = (
    "circle",
    "square",
    "diamond",
    "triangle-up",
    "pentagon",
    "hexagon",
    "star",
    "triangle-down",
)
_FILLS = ("", "-open", "-open-dot")


def marker(i: int, *, size: float = 8) -> dict[str, object]:
    """Marker style for scatter slot ``i``; unique for 8 × 8 × 3 = 192 slots.

    Colour and shape both have eight entries, so cycling them together would make slot 8
    identical to slot 0 in both. The shape therefore shifts by one on each colour wrap, and
    the fill (solid, open, open with dot) changes only after all 64 colour/shape pairs.

    Args:
        i: Zero-based series index.
        size: Marker size in px.

    Returns:
        A plotly ``marker`` dict with ``symbol``, ``size`` and ``color``.
    """
    n = len(_current.categorical)
    wrap = i // n
    shape = _SHAPES[(i + wrap) % len(_SHAPES)]
    fill = _FILLS[(i // (n * len(_SHAPES))) % len(_FILLS)]
    return {"symbol": shape + fill, "size": size, "color": color(i)}


def sequential_scale() -> list[list[object]]:
    """Plotly colorscale for the current sequential ramp (one hue, pale → deep)."""
    seq = _current.sequential
    return [[i / (len(seq) - 1), c] for i, c in enumerate(seq)]


def diverging_scale() -> list[list[object]]:
    """Plotly colorscale for the current diverging ramp (two poles, neutral midpoint)."""
    div = _current.diverging
    return [[i / (len(div) - 1), c] for i, c in enumerate(div)]
