# Changelog

## 0.1.0

First release.

- Theme: 8-hue colour-blind-validated categorical palette, sequential and diverging ramps, light and dark, installed as a plotly template (`set_style`, `use_style`).
- Layout: `finish` (title, subtitle, legend placement in pixels, margins), `apply_log` / `log_ticks`.
- Notebook helpers: `show`, `show_all`, `autoshow`; `save_html`, `save_png`.
- Time series, analysis, views, distributions, correlogram, matrices, comparisons, categorical charts, and `plotters.static` print figures.
- Input conventions A to E with `ConventionError` for the wrong one.
- Forecast evaluation: `plot_forecast`, `plot_residuals`, `plot_calibration`, `plot_quantile_calibration`, `plot_pit`, `plot_pacf` (and `plot_acf_grid(partial=True)`), `ljung_box`, `pacf_values`.
- Statistical learning: `plot_decision_boundary`, `plot_coefficient_path`, `plot_error_curves` (with the one-standard-error rule), `plot_roc`, `partial_dependence` / `plot_partial_dependence`, `plot_scree`, `plot_confusion`, `plot_dendrogram`; `plot_density2d` for large scatters.
- Small tasks: shaded `spans=` and `step=` on `plot_series`; `size=` (bubble) on `plot_scatter` and `plot_shaded_scatter`; `error=` on `plot_bars` and `plot_grouped_bars`; `plot_pareto`; `plot_grid` to combine finished figures.
- Diagnostics: `plot_gaps`, `plot_seasonal`, `plot_lag`, `plot_ccf`, `plot_periodogram`; `plot_bump`, `plot_table`, `plot_series(range_buttons=)`, `group_summary`.
- Reorganised for readability: new `relationships` module (scatter family), `comparisons` reduced to model comparison, shared checks in `_checks`; README restructured around a task-to-function table; gallery sections mirror the README.
- Every plotly plotter takes `width=` and `height=` (as in `plotly.express`); a test calls all 63 with both.
- `plot_views` keeps a series' colour across views (same trace name, same colour); `plot_series(max_drawn=)` (was `max_visible`).
- Visual gallery (`python -m tests.visual`) and a unit test suite.
