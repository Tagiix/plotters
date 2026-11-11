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
- `plot_views` keeps a series' colour across views (same trace name, same colour); `plot_series(drawn=)` (was `max_visible`).
- Fixes: `plot_series(drawn=0)` means the default instead of hiding every series; isolated points (a lone value between gaps) get a marker; `label=` on `plot_scatter` / `plot_shaded_scatter` is written on the plot only up to 40 points and is always on hover (hover is `closest`, not `x unified`); `plot_shaded_scatter` raises a clear error for `log_shade` with `reference <= 0`; `plot_null_test(observed_label=)`.
- `plot_shaded_scatter`: a `reference` outside the data (e.g. 0 for all-positive values, or non-positive under `log_shade`) falls back to the sequential ramp instead of one flat colour or an error; `clip=` keeps outliers from squashing the colour range. `plot_null_test`: rejection region as full-height bands with dotted cut lines, the observed value in its label, a default subtitle with the replicate count.
- Palette: 15 categorical colours (the original 8 unchanged, 7 added by a colour-blind-separation search), `set_style(colors=8..15)` / `use_style(colors=)` to pick fewer; markers stay unique at any size. `plot_cumulative`, `plot_drawdown` and `plot_rolling` lose the 8-series cap and take `drawn=` like `plot_series`.
- `plot_tradeoff(frontier=True)`: set `False` for the plain scatter without the Pareto frontier line and rings (default unchanged).
- **No more colour-count refusals.** Every chart that raised "draws at most 3 / 8" now draws (colour repeats past the palette, shape/dash change); `MAX_ALL_PAIRS` is removed. `label=` is written once per distinct value and always on hover; `plot_tradeoff` / `plot_scatter` groups stay distinct by colour and shape at any count.
- `analyze_series(hist_range=)` forwards the histogram drawing window to `plot_series_distribution`.
- `series_statistics` gains `rms`, `mean_absolute`, `mean_over_std`, `median` and the options `tail_normalize`, `tail_center`, `tail_ratios`, `energy_fractions`, `acf_lags`, `rolling_window`, `turnover`. `analyze_series` gains `reference_series`, `reference_label`, `normal_reference`, `turnover_ratio`, `value_label`, `view_prefix` and passes the tail options through. `plot_null_test(normal_reference=, null_label=)` adds an N(0, 1) line and a legend.
- `analyze_series` returns the full statistics table by default: `tail_normalize="rms"`, `thresholds=(1, 2, 3, 4)`, `tail_ratios=((3, 1),)`, `energy_fractions=(0.01,)`, `acf_lags=(1, 5, 21)`, `rolling_window=21`, `extreme_multiples=(3, 4)`, quantiles 1/5/50/95/99. Quantile columns are now zero-padded (`q01`, `q05`, `q50`, `q95`, `q99`; was `q1`, `q5`). `series_statistics` gains `extreme_multiples`.
- `analyze_series(readable_columns=True)` titles the derived columns for reading (`P(|z| > 1)`, `n(|z| > 1)`, `P(|z| > 3) / P(|z| > 1)`, `Top 1% squared share`, `ACF lag 5`, `Std / mean of 21-obs rolling RMS` ...); `False` keeps the code names. The result also carries the old keys `tail_ratios`, `tail_counts`, `energy_shares`, `normal_tail_probabilities`, `normal_tail_ratios`, `turnover_ratio`, `pooled`, `per_series` / `per_ticker`, `histories`, `normalization`, `n_series` / `n_tickers`, `skipped_series`, `pooling`.
- `plot_tradeoff(rings=, label_text=)`: no rings on the frontier, labels decluttered (`"auto"`), every label, or hover only; hover is `closest`.
- Visual gallery (`python -m tests.visual`) and a unit test suite.
