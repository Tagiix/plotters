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
- Visual gallery (`python -m tests.visual`) and a unit test suite.
