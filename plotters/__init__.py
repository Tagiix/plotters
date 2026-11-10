"""plotters: generic pandas in, themed plotly ``Figure`` out.

Nothing here calls ``.show()``; the caller decides. Call :func:`set_style` once at the top.
Print figures live in :mod:`plotters.static`, which needs matplotlib and is not imported here.
"""

from importlib.metadata import PackageNotFoundError, version

from ._errors import ConventionError
from ._types import MAX_ALL_PAIRS, MAX_SERIES, MAX_VISIBLE, SeriesLike
from .analysis import analyze_series, compare_series, plot_series_distribution
from .comparisons import (
    plot_box_series,
    plot_interval,
    plot_paired_difference,
    plot_rank_heatmap,
    plot_scatter,
    plot_shaded_scatter,
    plot_small_multiples,
    plot_tradeoff,
)
from .correlogram import plot_acf, plot_acf_grid
from .distributions import (
    plot_box,
    plot_cumulative_share,
    plot_ecdf,
    plot_expectile_curve,
    plot_histogram,
    plot_null_test,
    plot_qq,
)
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
from .matrices import plot_contours, plot_correlation, plot_matrix, plot_table_heatmap
from .notebook import autoshow, show, show_all
from .stats import (
    acf_values,
    bartlett_band,
    normalise_series,
    normalize_series,
    sample_expectile,
    series_statistics,
    series_turnover_ratio,
)
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
from .views import annotate_views, plot_views

try:
    __version__ = version("plotters")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = [
    "plot_box_series",
    "plot_interval",
    "plot_paired_difference",
    "plot_rank_heatmap",
    "plot_scatter",
    "plot_shaded_scatter",
    "plot_small_multiples",
    "plot_tradeoff",
    "plot_contours",
    "plot_correlation",
    "plot_matrix",
    "plot_table_heatmap",
    "plot_acf",
    "plot_acf_grid",
    "plot_box",
    "plot_cumulative_share",
    "plot_ecdf",
    "plot_expectile_curve",
    "plot_histogram",
    "plot_null_test",
    "plot_qq",
    "analyze_series",
    "compare_series",
    "plot_series_distribution",
    "acf_values",
    "bartlett_band",
    "normalise_series",
    "normalize_series",
    "sample_expectile",
    "series_statistics",
    "series_turnover_ratio",
    "annotate_views",
    "plot_views",
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
