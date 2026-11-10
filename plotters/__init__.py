"""plotters: generic pandas in, themed plotly ``Figure`` out.

Nothing here calls ``.show()``; the caller decides. Call :func:`set_style` once at the top.
Print figures live in :mod:`plotters.static`, which needs matplotlib and is not imported here.
"""

from importlib.metadata import PackageNotFoundError, version

from ._errors import ConventionError
from ._types import MAX_ALL_PAIRS, MAX_SERIES, MAX_VISIBLE, SeriesLike
from .frames import (
    as_series_dict,
    group_by_prefix,
    group_values,
    label_from,
    long_to_wide,
    normalise_within,
    normalize_within,
    pick,
)
from .io import save_html, save_png
from .layout import apply_log, finish, log_ticks
from .notebook import autoshow, show, show_all
from .style import (
    color,
    colour,
    diverging_scale,
    get_style,
    line_style,
    marker,
    palette,
    sequential_scale,
    set_style,
    use_style,
)
from .timeseries import (
    plot_bars_with_line,
    plot_cumulative,
    plot_drawdown,
    plot_rolling,
    plot_series,
    plot_stacked,
)

try:
    __version__ = version("plotters")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = [
    "ConventionError",
    "MAX_ALL_PAIRS",
    "MAX_SERIES",
    "MAX_VISIBLE",
    "SeriesLike",
    "as_series_dict",
    "group_by_prefix",
    "group_values",
    "label_from",
    "long_to_wide",
    "normalise_within",
    "normalize_within",
    "pick",
    "save_html",
    "save_png",
    "apply_log",
    "finish",
    "log_ticks",
    "autoshow",
    "show",
    "show_all",
    "color",
    "colour",
    "diverging_scale",
    "get_style",
    "line_style",
    "marker",
    "palette",
    "sequential_scale",
    "set_style",
    "use_style",
    "plot_bars_with_line",
    "plot_cumulative",
    "plot_drawdown",
    "plot_rolling",
    "plot_series",
    "plot_stacked",
]
