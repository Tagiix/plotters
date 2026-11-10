"""plotters: generic pandas in, themed plotly ``Figure`` out.

Nothing here calls ``.show()``; the caller decides. Call :func:`set_style` once at the top.
Print figures live in :mod:`plotters.static`, which needs matplotlib and is not imported here.
"""

from importlib.metadata import PackageNotFoundError, version

from ._errors import ConventionError
from ._types import MAX_SERIES, MAX_VISIBLE, SeriesLike
from .analysis import (
    analyze_series,
    compare_series,
    plot_series_distribution,
)
from .categorical import (
    plot_bars,
    plot_counts,
    plot_dumbbell,
    plot_grouped_bars,
    plot_lollipop,
    plot_pareto,
    plot_waterfall,
)
from .comparisons import (
    plot_box_series,
    plot_bump,
    plot_interval,
    plot_paired_difference,
    plot_rank_heatmap,
    plot_small_multiples,
    plot_tradeoff,
)
from .correlogram import (
    plot_acf,
    plot_acf_grid,
    plot_pacf,
)
from .diagnostics import (
    plot_ccf,
    plot_gaps,
    plot_lag,
    plot_periodogram,
    plot_seasonal,
)
from .distributions import (
    plot_box,
    plot_cumulative_share,
    plot_ecdf,
    plot_expectile_curve,
    plot_histogram,
    plot_null_test,
    plot_qq,
)
from .forecasting import (
    plot_calibration,
    plot_forecast,
    plot_pit,
    plot_quantile_calibration,
    plot_residuals,
)
from .frames import (
    as_series_dict,
    as_value_series,
    group_by_prefix,
    group_summary,
    group_values,
    label_from,
    long_to_wide,
    normalise_within,
    normalize_within,
    pick,
)
from .io import (
    save_html,
    save_png,
)
from .layout import (
    HEIGHT,
    apply_log,
    finish,
    log_ticks,
)
from .learning import (
    partial_dependence,
    plot_coefficient_path,
    plot_confusion,
    plot_decision_boundary,
    plot_dendrogram,
    plot_error_curves,
    plot_partial_dependence,
    plot_roc,
    plot_scree,
)
from .matrices import (
    plot_contours,
    plot_correlation,
    plot_matrix,
    plot_table,
    plot_table_heatmap,
)
from .notebook import (
    autoshow,
    show,
    show_all,
)
from .relationships import (
    plot_density2d,
    plot_regression,
    plot_scatter,
    plot_scatter_matrix,
    plot_shaded_scatter,
)
from .stats import (
    Normalized,
    acf_values,
    average_precision,
    bartlett_band,
    ccf_values,
    ljung_box,
    normalise_series,
    normalize_series,
    one_standard_error_choice,
    pacf_values,
    periodogram,
    precision_recall_curve,
    roc_auc,
    roc_curve,
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
    plot_calendar_heatmap,
    plot_cumulative,
    plot_drawdown,
    plot_line_band,
    plot_rolling,
    plot_series,
    plot_stacked,
)
from .views import (
    annotate_views,
    plot_grid,
    plot_views,
)

try:
    __version__ = version("plotters")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    # Errors, limits and types
    "ConventionError",
    "MAX_SERIES",
    "MAX_VISIBLE",
    "SeriesLike",
    # Theme and layout
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
    "HEIGHT",
    "apply_log",
    "finish",
    "log_ticks",
    "autoshow",
    "show",
    "show_all",
    "save_html",
    "save_png",
    # Data shaping and statistics
    "as_series_dict",
    "as_value_series",
    "group_by_prefix",
    "group_summary",
    "group_values",
    "label_from",
    "long_to_wide",
    "normalise_within",
    "normalize_within",
    "pick",
    "Normalized",
    "acf_values",
    "average_precision",
    "bartlett_band",
    "ccf_values",
    "ljung_box",
    "normalise_series",
    "normalize_series",
    "one_standard_error_choice",
    "pacf_values",
    "periodogram",
    "precision_recall_curve",
    "roc_auc",
    "roc_curve",
    "sample_expectile",
    "series_statistics",
    "series_turnover_ratio",
    # Time series
    "plot_bars_with_line",
    "plot_calendar_heatmap",
    "plot_cumulative",
    "plot_drawdown",
    "plot_line_band",
    "plot_rolling",
    "plot_series",
    "plot_stacked",
    "analyze_series",
    "compare_series",
    "plot_series_distribution",
    "plot_ccf",
    "plot_gaps",
    "plot_lag",
    "plot_periodogram",
    "plot_seasonal",
    "plot_acf",
    "plot_acf_grid",
    "plot_pacf",
    "plot_calibration",
    "plot_forecast",
    "plot_pit",
    "plot_quantile_calibration",
    "plot_residuals",
    # Distributions
    "plot_box",
    "plot_cumulative_share",
    "plot_ecdf",
    "plot_expectile_curve",
    "plot_histogram",
    "plot_null_test",
    "plot_qq",
    # Relationships between columns
    "plot_density2d",
    "plot_regression",
    "plot_scatter",
    "plot_scatter_matrix",
    "plot_shaded_scatter",
    # Comparing models and candidates
    "plot_box_series",
    "plot_bump",
    "plot_interval",
    "plot_paired_difference",
    "plot_rank_heatmap",
    "plot_small_multiples",
    "plot_tradeoff",
    # Categories, counts, matrices and tables
    "plot_bars",
    "plot_counts",
    "plot_dumbbell",
    "plot_grouped_bars",
    "plot_lollipop",
    "plot_pareto",
    "plot_waterfall",
    "plot_contours",
    "plot_correlation",
    "plot_matrix",
    "plot_table",
    "plot_table_heatmap",
    # Statistical learning
    "partial_dependence",
    "plot_coefficient_path",
    "plot_confusion",
    "plot_decision_boundary",
    "plot_dendrogram",
    "plot_error_curves",
    "plot_partial_dependence",
    "plot_roc",
    "plot_scree",
    # Combining figures
    "annotate_views",
    "plot_grid",
    "plot_views",
]
