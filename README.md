# plotters

Generic plotly charts with one consistent, colour-blind-safe look. Pandas in, a themed plotly `Figure` out.

Nothing calls `.show()`: the caller decides. Analysis helpers also return the statistics and series behind their figures. It draws and summarises what you give it; it never fits a model for you.

```python
import plotters as pl

pl.set_style(dark=False)                                  # once, at the top

pl.plot_series(returns, "Daily returns", percent=True)    # one line per column
pl.plot_bars({"cats": 42, "dogs": 31, "birds": 12})       # straight from a dict of counts
pl.plot_box(by_group, reference=1.0)                      # rank groups by median
pl.plot_tradeoff(table, x="cost", y="error", label="model", logx=True)
```

## See every chart

The repository includes a gallery of every function, in light and dark side by side. Generate and open it from a checkout (the library installed with `[dev]`, see [Install](#install)):

```bash
cd ~/code/plotters
python -m tests.visual          # writes gallery/index.html  (add a word to build only matching cards, e.g. "views")
open gallery/index.html         # macOS; on Linux use xdg-open, on Windows start
```

Keep `gallery/index.html` next to `gallery/plotly.min.js` (it needs it), and zip the folder to share it. It is not committed to git: it is regenerated on demand.

## Contents

1. [Install](#install)
2. [Getting started](#getting-started): theme, notebooks, saving, hover docs
3. [Which function do I need?](#which-function-do-i-need): a task-to-function table
4. [What to pass](#what-to-pass-five-input-conventions): the five input shapes
5. [Chart reference](#chart-reference): one section per module
6. [Design rules](#design-rules-this-library-enforces) and [writing your own plotter](#writing-your-own-plotter)
7. [Development](#development): gallery, tests, type checking, project map

## Install

From GitHub into the active Python environment (the SSH form also works while the repository is private):

```bash
pip install "plotters @ git+ssh://git@github.com/Tagiix/plotters.git"                 # core
pip install "plotters[stats,static] @ git+ssh://git@github.com/Tagiix/plotters.git"    # + scipy, matplotlib
pip install "plotters @ git+ssh://git@github.com/Tagiix/plotters.git@v0.1.0"          # a pinned release
pip install -U --force-reinstall --no-deps "plotters @ git+ssh://git@github.com/Tagiix/plotters.git"  # upgrade
```

With `uv`, into a project: `uv add "plotters @ git+ssh://git@github.com/Tagiix/plotters.git"`. Or list it in your project's `pyproject.toml` (or `requirements.txt` without the `dependencies = [...]` wrapper):

```toml
dependencies = ["plotters @ git+ssh://git@github.com/Tagiix/plotters.git@v0.1.0"]
```

Once the repository is public, `git+https://github.com/Tagiix/plotters.git` works as well.

Core needs `pandas`, `numpy`, `plotly` and nothing else. `scipy` is only for `plot_qq(dist="t")`, `plot_dendrogram` and the p value of `ljung_box` (imported lazily; extra `stats`), and `matplotlib` only for `plotters.static` (extra `static`). The package ships type information (`py.typed`), so hover docs and completion work in your other projects too.

**One clone, many projects (recommended for your own machine).** Keep a single checkout and install it *editable* into each project's venv. Every project then imports straight from that folder, and one `git pull` updates all of them at once:

```bash
git clone git@github.com:Tagiix/plotters.git ~/code/plotters      # once per machine

# install into any venvs (extras via PLOTTERS_EXTRAS: stats = scipy, static = matplotlib)
PLOTTERS_EXTRAS=stats,static ~/code/plotters/scripts/install-into.sh ~/code/project-a/.venv
PLOTTERS_EXTRAS=static       ~/code/plotters/scripts/install-into.sh ~/code/project-b/.venv

git -C ~/code/plotters pull                                       # update every project
```

The script (`scripts/install-into.sh`) takes any number of venv folders, uses `uv` if present and `pip` otherwise, and prints where each environment imports `plotters` from. It never edits the projects themselves, so their `requirements.txt` stays machine-independent.

**Developing the library** (the same editable install, plus the test tools):

```bash
cd ~/code/plotters && uv venv && uv pip install -e ".[dev]"      # or: python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

The build backend (hatchling) writes a plain static path for editable installs, so Pylance and pyright resolve the import with no extra flag.

## Getting started

**Theme.** `set_style(dark=False)` chooses the palette and installs a plotly template (`colors=` picks how many of the 15 categorical colours to cycle through, from 8 to 15), so even a plain `go.Figure()` created afterwards gets the fonts, colours and grid. `use_style(dark=True)` is a context manager that restores the previous theme. `set_style("auto")` reads the `PLOTTERS_STYLE` environment variable.

**Showing figures.** These functions return a Figure and draw nothing, so Jupyter renders only the **last expression in a cell**. Three ways around it:

```python
pl.show(fig_a, fig_b)                 # explicit, any number
pl.show_all(out["figures"])           # a dict or list of figures
pl.autoshow(True)                     # every plotter displays as soon as it finishes, mid-cell
```

`autoshow` hooks `finish()`, so it also works for your own plotters. Assign the result (`fig = pl.plot_series(...)`) or end the line with `;` to avoid Jupyter showing it twice.

**Saving.**

```python
pl.save_html(fig, "chart.html")                 # keeps interactivity, self-contained
pl.save_html(fig, "chart.html", cdn=True)       # ~3 MB smaller, needs internet
pl.save_png(fig, "chart.png")                   # needs kaleido and Chrome (plotly_get_chrome)
```

`write_image` needs kaleido **and** a Chrome install; on a headless box it usually fails. For PNGs without Chrome (or print figures at an exact size) use [`plotters.static`](#print-figures-plottersstatic).

**Sizing.** Like `plotly.express`, every plotter takes `height=` and `width=` in pixels, so `pl.plot_series(df, width=640, height=360)` works on any chart. `width=None` (the default) fills the container, which is what you want in a notebook or an exported page. Grid-style charts (`plot_small_multiples`, `plot_acf_grid`, `plot_lag`, `plot_grid`, ...) compute their height from the number of panels, and `height=` overrides that. A test calls all 63 plotters with both arguments, so this stays true.

**Hover documentation.** Every public function has a docstring listing its arguments, defaults and what it raises, and `Literal[...]` types show the allowed values. In VS Code, hover a function to read it.

## Which function do I need?

| I want to ... | Use |
|---|---|
| **Time series** | |
| plot one or many lines (percent, log, area, step, slider, date-range buttons) | `plot_series` |
| mark events, outages or recessions on a line chart | `plot_series(markers=, spans=, bands=)` |
| switch between subsets of the lines with a dropdown | `plot_series(views=...)` |
| compare realised bars with predicted lines | `plot_bars_with_line` |
| see cumulative growth, drawdown, shares over time, a rolling statistic | `plot_cumulative`, `plot_drawdown`, `plot_stacked`, `plot_rolling` |
| draw a line with an uncertainty ribbon | `plot_line_band` |
| see a daily series as a calendar | `plot_calendar_heatmap` |
| check where data is missing | `plot_gaps` |
| see seasonality (one line per year / week / day) | `plot_seasonal` |
| check if a value resembles an earlier one | `plot_lag`, `plot_acf`, `plot_pacf` |
| see whether one series leads another | `plot_ccf` |
| find cycles | `plot_periodogram` |
| analyse a set of series in one call (stats + 5 figures) | `analyze_series` |
| compare series against one reference | `compare_series`, `plot_paired_difference` |
| **Forecasts and models** | |
| show a forecast with its interval | `plot_forecast` |
| judge a model's residuals | `plot_residuals` |
| check if probabilities / quantiles / distributions are honest | `plot_calibration`, `plot_quantile_calibration`, `plot_pit` |
| trade off two objectives (Pareto frontier) | `plot_tradeoff` |
| rank candidates across contexts or over time | `plot_rank_heatmap`, `plot_bump` |
| show estimates with intervals | `plot_interval` (+ `group_summary` to compute them) |
| draw ROC / precision-recall, a decision boundary, error vs complexity, coefficient paths | `plot_roc`, `plot_decision_boundary`, `plot_error_curves`, `plot_coefficient_path` |
| **Distributions** | |
| see a distribution | `plot_histogram`, `plot_ecdf`, `plot_box` |
| compare to a normal or Student-t | `plot_qq` |
| show how concentrated a total is | `plot_cumulative_share`, `plot_pareto` |
| picture a hypothesis test | `plot_null_test` |
| **Two numeric columns** | |
| scatter (grouped, labelled, bubble-sized, coloured by a third) | `plot_scatter`, `plot_shaded_scatter` |
| scatter with a lot of points | `plot_density2d` |
| scatter with a fitted line and band | `plot_regression` |
| pairwise scatter of several columns | `plot_scatter_matrix` |
| **Categories** | |
| bars from a dict of counts or a Series | `plot_bars`, `plot_counts` |
| bars with error bars | `plot_bars(error=)`, `plot_grouped_bars(error=)` |
| compare a few series across categories | `plot_grouped_bars` |
| show shares (instead of a pie) | `plot_bars(normalize=True)`, `plot_grouped_bars(normalize=True)` |
| many categories, or negative values | `plot_lollipop` |
| before vs after per item | `plot_dumbbell` |
| a bridge from a start to a total | `plot_waterfall` |
| **Tables and matrices** | |
| show a DataFrame as a table | `plot_table` |
| a matrix coloured by its values | `plot_matrix`, `plot_table_heatmap` |
| correlations between columns | `plot_correlation` |
| a function of two variables | `plot_contours` |
| a confusion matrix, PCA scree plot, dendrogram | `plot_confusion`, `plot_scree`, `plot_dendrogram` |
| **Layout** | |
| one panel per value of a column | `plot_small_multiples` |
| put several finished figures in a grid | `plot_grid` |
| switch between finished figures with a dropdown | `plot_views`, `annotate_views` |
| **Reshaping and numbers** | |
| long to wide, group a column, build labels | `long_to_wide`, `group_values`, `group_by_prefix`, `label_from`, `normalize_within` |
| summary statistics, ACF/PACF, Ljung-Box, ROC AUC, expectiles | `series_statistics`, `acf_values`, `pacf_values`, `ljung_box`, `roc_auc`, `sample_expectile` |

If a task is not here, [write your own plotter](#writing-your-own-plotter) in a dozen lines and keep the same look.

## What to pass: five input conventions

The first argument tells you which one a plotter uses. Picking the wrong one is the most common error, so the library names the convention in a `ConventionError` instead of failing deep inside pandas.

| Convention | First argument | Used by |
|---|---|---|
| **A. Series-like** | `data`: a Series, a **wide** DataFrame (one series per column), or `{label: Series}` | time series, distributions, correlogram, analysis |
| **B. Long frame + column names** | `frame`: a **long/tidy** DataFrame, then `x=`, `y=`, `label=` ... as *column name* strings | relationships, comparisons |
| **C. Two positional series** | `bars` (a Series) and `line` (Series or dict) | `plot_bars_with_line`, `plot_forecast`, `plot_ccf` |
| **D. Wide / square frame as-is** | `frame`: rows and columns are the two axes, cells are the values | matrices, `plot_grouped_bars`, `plot_bump` |
| **E. Label → number** | a `{label: number}` dict, or a Series indexed by label (`value_counts()` works) | `plot_bars`, `plot_lollipop`, `plot_waterfall`, `plot_pareto` |

```python
pl.plot_scatter(frame_wide)            # ConventionError: plot_scatter expects a long (tidy) DataFrame
                                       # followed by column-name strings (convention B); got dict.
```

Convert between shapes with `long_to_wide`, `group_values`, `as_series_dict`, `as_value_series`.

**Subsetting and colour order.** Convention-A plotters take `series=[...]`, which both picks a subset *and* fixes the order, and the order assigns the colour slots. An unknown name raises `KeyError: "['zz'] not in ['a', 'b', 'c']"`. It is `series=` everywhere, including `plot_box`.

**`group=` and `label=` differ.** `group="model"` splits into one coloured series per value (at most 3, since scatter is read pair by pair); `label="model"` keeps one series and writes the text next to each point. `groups={family: prefixes}` collapses many members into families (the longest prefix wins) so colour means family and shape means member.

## Chart reference

One section per module. Everything is imported from `plotters` directly (`pl.plot_series`); the module name only says where to look in the source.

### Time series: `timeseries`

| Function | For |
|---|---|
| `plot_series` | One line per series: the workhorse. `percent`, `logy`, `logx`, `slider`, `range_buttons`, `fill`, `step`, `bands` (horizontal reference lines), `markers` (vertical lines), `spans` (shaded bands such as a recession or outage), `views` (dropdown over named subsets), `drawn` |
| `plot_bars_with_line` | Realised as bars, prediction as lines on one axis. Did it track or lag? |
| `plot_cumulative` | Wealth curve from per-period values; `compound=True` for reinvested; `drawn` |
| `plot_drawdown` | Drop from the running peak, filled because the area *is* the quantity; `drawn` |
| `plot_stacked` | Parts of a whole; refuses negative parts; `normalize=True` for shares |
| `plot_rolling` | A rolling statistic over the raw series in lighter ink; `drawn` |
| `plot_line_band` | A line with an uncertainty ribbon |
| `plot_calendar_heatmap` | A daily series as a weekday-by-week grid |

`plot_series` has **no series limit**: every series is kept but only the first `drawn` (default 20) start drawn, the rest sit greyed out in the legend, one click to add. With `views={"A": [...], "B": [...]}` a dropdown swaps named subsets of the lines already passed (each series is drawn once and the menu toggles `visible`, so an exported HTML keeps working with no server). A series named in no view stays visible in all of them, so a pooled baseline sits behind each per-ticker view without repeating it. Past 8 such series the views become an explicit allowlist.

### Statistics and analysis: `analysis`

| Function | For |
|---|---|
| `analyze_series` | Statistics plus five figures (values, absolute values, cumulative sum, cumulative absolute sum, histogram), each with a pooled view and one view per series |
| `compare_series` | Several series on the same dates against one reference (`align="inner"/"outer"`); differences are reference minus the others |
| `plot_series_distribution` | Histogram with a dropdown, statistics block and quantile markers |

```python
out = pl.analyze_series(returns, bins=100, normalize="std", quantiles=(0.01, 0.5, 0.99))
out["stats"]                          # DataFrame
pl.show_all(out["figures"])           # every figure from one notebook cell
```

A pooled **distribution** and a pooled **path** are different quantities: the histogram stacks observations with equal weight per observation, while the time plots average the series available at each index value. Set `pooled=False` when combining the inputs' units makes no sense. Histogram limits (`hist_range`) never erase tail probability: values outside the window stay in the density denominator. `normal_reference=True` overlays N(0, 1), not a fitted normal.

### Diagnostics: `diagnostics`

| Function | For |
|---|---|
| `plot_gaps` | Where each series has missing data: a row per series, shaded by the share missing in each stretch of time |
| `plot_seasonal` | One line per year (or ISO week, or day) laid over Jan-Dec (or Mon-Sun, or 0-23h); the dashed line is the average |
| `plot_lag` | Each value against the one k steps earlier, one panel per lag, correlation in the title |
| `plot_ccf` | Cross-correlation in both directions: a peak at a negative lag means the second series follows the first |
| `plot_periodogram` | Variance by frequency (or period); the largest spikes are labelled with their period |

### Autocorrelation: `correlogram`

`plot_acf` draws a stem from zero to each coefficient, a marker at the tip, and the significance band as a ribbon. **The band widens with the lag**, and that is the point: Bartlett's formula inherits the autocorrelation already present at shorter lags and equals the textbook `1.96/√n` only at lag 1. `absolute=True` plots `acf(|x|)`: on a return series `acf(x)` asks whether the sign is predictable, `acf(|x|)` whether the magnitude is (volatility clustering, the ARCH case).

`plot_pacf` is the partial autocorrelation: for an AR(p) series it cuts off after lag p where the acf only decays, and its band is the flat `z/√n`. `plot_acf_grid` gives each series its own panel and band with no series cap (`partial=True` for the pacf).

### Forecast evaluation: `forecasting`

| Function | For |
|---|---|
| `plot_forecast` | History, forecast(s) and prediction interval with the forecast origin marked; `actual=` draws what really happened |
| `plot_residuals` | Four panels: over time (or against `fitted=`), distribution vs a normal, normal QQ, acf with band; the subtitle gives moments and Ljung-Box |
| `plot_calibration` | Reliability diagram with Wilson intervals: when a forecast says 70%, how often does it happen? A curve **flatter** than the diagonal means overconfident |
| `plot_quantile_calibration` | Observed fraction below each forecast quantile vs its nominal level, with the ribbon sampling noise allows |
| `plot_pit` | Probability-integral-transform histogram: U = overconfident, hump = too wide, slope = biased |

### Distributions: `distributions`

| Function | For |
|---|---|
| `plot_null_test` | A simulated distribution with the tested value and the rejection region shaded: the picture of a hypothesis test. The p value is counted from the distribution |
| `plot_box` | One box per group ranked by median: the ranking chart. `points=True`, `reference=`, `logvalue=` |
| `plot_histogram` | Overlaid distributions on shared bins, optional normal fit |
| `plot_ecdf` | The distribution with no bin-width choice; `complementary=True` with `logy=True` for tails |
| `plot_qq` | Against the normal, or `dist="t"` for Student-t with ν fitted by maximum likelihood (needs scipy). Reference line through the IQR |
| `plot_cumulative_share` | How concentrated a total is, against a "paid evenly" diagonal |
| `plot_expectile_curve` | The tau expectile over each series' own mean against tau. The slope is the reading |

`plot_qq(dist="t")` is the one to reach for on fat tails: a normal QQ bends at both ends and stops there, saying the normal is wrong but not what is right. On a simulated t(5) sample the fit recovers ν ≈ 5.

### Relationships between columns: `relationships`

| Function | For |
|---|---|
| `plot_scatter` | Two columns, optional `group=`, `label=`, `y = x` diagonal, and `size=` for a bubble chart (marker **area** tracks the value) |
| `plot_shaded_scatter` | Two columns plus a continuous third as colour with a colourbar; `size=` adds a fourth variable |
| `plot_density2d` | A binned density for scatters with many points: cells shaded by count (log-scaled by default), empty cells blank |
| `plot_regression` | Scatter with an OLS fit and a 95% band for the mean (numpy only) |
| `plot_scatter_matrix` | Pairwise scatter of up to 6 columns with diagonal histograms |

### Comparing models and candidates: `comparisons`

| Function | For |
|---|---|
| `plot_tradeoff` | Two competing objectives and the Pareto frontier; `lower_is_better`, `groups=`, `connect=` |
| `plot_rank_heatmap` | Items by contexts coloured by rank within each context (the one chart that colours by rank) |
| `plot_bump` | Rank of each item at every step over time, names at the right end; `highlight=` greys out the rest |
| `plot_interval` | One horizontal interval per row; intervals crossing `reference` are hollow (not separable) |
| `plot_box_series` | A box at each numeric `x`, medians joined into a curve per group |
| `plot_small_multiples` | One panel per value of `panel`, shared legend |
| `plot_paired_difference` | Cumulative difference against a baseline series (`base` names a series) |

`plot_box_series` needs a **numeric x**, and that is not a style choice: a plotly `Box` width is in data units, so the same width renders many times wider at one end of a log axis than the other. The axis is linear, ticked at the values that exist, and the width comes from the tightest gap divided by the group count. `lower_is_better` is a bool or a `(x_flag, y_flag)` pair wherever two axes exist. For estimates with intervals, `group_summary(frame, by, value)` computes a mean or median and its confidence interval per group, ready for `plot_interval`.

### Categories and counts: `categorical`

| Function | For |
|---|---|
| `plot_bars` | Bars from a `{label: number}` dict or a Series, value written at each tip. `sort`, `top`, `normalize`, `error=` (symmetric, or a `(minus, plus)` pair) |
| `plot_counts` | `value_counts` and the bars in one call |
| `plot_grouped_bars` | A few series across categories; `stacked`, `normalize`, `error=` |
| `plot_lollipop` | Stems and dots: less ink, good for many categories or negative values |
| `plot_dumbbell` | Before and after per item |
| `plot_waterfall` | Contributions adding up to a total |
| `plot_pareto` | Causes ranked by share with the running total and an 80% line; `top=` folds the tail into "Other" |

### Matrices and tables: `matrices`

Colour by **value** (a rank would discard magnitude).

| Function | For |
|---|---|
| `plot_matrix` | Any wide/square frame. A frame straddling `center` gets the diverging ramp with a symmetric range, otherwise the single-hue ramp |
| `plot_correlation` | Correlation between columns (takes the **data**), diverging about 0, diagonal masked; `method="spearman"` for fat tails |
| `plot_table_heatmap` | A table whose columns are on different scales (a moments table); each column shaded on its own range, actual values written in |
| `plot_contours` | A function of two variables as labelled level curves, from a grid you evaluated yourself |
| `plot_table` | A DataFrame as a plain styled table: aligned, formatted per column, missing as a dash |

Cell text is coloured from the **measured luminance** of each fill, never from a position threshold: the dark-mode diverging ramp has a dark midpoint, so a threshold rule puts dark text on dark fill.

### Statistical learning: `learning`

Plots for the model-fitting workflow (the kind drawn throughout *The Elements of Statistical Learning*). They draw what you have computed; none fits a model.

| Function | For |
|---|---|
| `plot_decision_boundary` | A classifier's boundary from a grid of scores, tinted sides, training points, and a dashed reference boundary (e.g. Bayes) |
| `plot_coefficient_path` | Ridge / lasso profiles against λ or degrees of freedom, names at the line ends |
| `plot_error_curves` | Train / test / CV error against complexity with standard-error ribbons, a reference level, and the minimum or the **one-standard-error** choice marked |
| `plot_roc` | ROC or precision-recall curves with AUC / AP in the legend |
| `partial_dependence`, `plot_partial_dependence` | Compute and draw how a model's prediction responds to each feature, with a rug of data deciles. `partial_dependence` only calls the `predict` function you pass |
| `plot_scree` | Variance explained by each principal component with the running total |
| `plot_confusion` | Confusion matrix, rows as shares of the actual class (hover shows counts) |
| `plot_dendrogram` | Hierarchical clustering tree with the cut for `n_clusters` (needs scipy) |

### Multiple views and combining figures: `views`

Four ways to show several things, depending on what differs between them:

| You want | Use |
|---|---|
| A dropdown over **subsets of the same lines** | `plot_series(..., views={"A": [...], "B": [...]})` |
| A dropdown over **different figures** (their own axes, annotations, shapes) | `plot_views({"Raw": fig_a, "Normalised": fig_b})` |
| A **changing note** (n, a statistic) that follows the dropdown | `annotate_views(fig, {view: text})` |
| Everything visible **at once**, side by side | `plot_grid({"Raw": fig_a, "Normalised": fig_b}, ncols=2)` |

`analyze_series` and `plot_series_distribution` already give one dropdown view per series plus a pooled one.

`plot_views` merges prepared figures into one; traces, axes, annotations, shapes and the legend switch together, and the first view opens initially. `plot_grid` is the static counterpart: each figure keeps its own axes, titles, shapes and annotations in its own panel, and a series name shared by several panels is listed in the legend once. Both are Cartesian-only: existing menus, sliders, frames and non-Cartesian traces raise rather than silently losing their interactions.

### Data shaping and numbers: `frames`, `stats`

Useful on their own, and the bridge between the input conventions.

| Function | Returns |
|---|---|
| `long_to_wide(frame, index, columns, values)` | Wide DataFrame. **Orders numeric columns numerically**, otherwise `pivot` sorts them as strings and `126` lands between `18` and `21` |
| `group_values(frame, by, value, keep=)` | `{group: Series}`, ordered by median unless `keep` fixes it |
| `group_summary(frame, by, value)` | A mean or median per group with a confidence interval (normal or bootstrap) |
| `normalize_within(frame, value, within, how="min")` | `value` divided by its own group's reference |
| `group_by_prefix(labels, {group: (prefix, ...)}, other=)` | Group names by the **longest** matching prefix; raises on an unmatched label unless `other` is given |
| `label_from(frame, *columns, sep=" ")` | Labels for `label=`; NaN parts are skipped |
| `as_series_dict`, `as_value_series`, `pick` | The convention normalisers, and subsetting in the order given |
| `series_statistics`, `series_turnover_ratio`, `normalize_series` | Moments, quantiles, tail frequencies, turnover; per-series scaling |
| `acf_values`, `pacf_values`, `bartlett_band`, `ccf_values`, `ljung_box`, `periodogram` | The numbers behind the time-series diagnostics |
| `sample_expectile`, `roc_curve`, `roc_auc`, `precision_recall_curve`, `average_precision`, `one_standard_error_choice` | The numbers behind the distribution and learning plots |

### Print figures: `plotters.static`

Not imported by `plotters/__init__.py`: it needs matplotlib. `from plotters import static as ps`.

```python
import plotters as pl
from plotters import static as ps

pl.set_style(dark=False)                 # same palette as the plotly charts
ps.use_print_style()                     # rcParams: serif, 9 pt, the palette's inks
figure = ps.plot_lines(frame, title="...", ylabel="...", label_last=True)
ps.save(figure, "figures/thing.png")     # 300 dpi, closes the figure
```

`plot_lines`, `plot_points` (with `frontier=True`), `plot_ranked_bars`, `plot_heatmap`, `plot_function`, `plot_stems` (a correlogram with a per-lag band) and `plot_panels`. `figure_size(width=0.86)` gives inches for that fraction of `\textwidth` at a4paper with 2.2 cm margins. **Size here, never with `includegraphics[scale=]`**: scaling in LaTeX scales the figure's text too.

What changes for print: `vary_dash` defaults ON (no legend to click, and four solid lines photocopy as one); end-of-line labels are nudged apart; subtitles wrap to the axes width; cell ink comes from measured luminance; `percent=True` writes a bare `%`.

## Theme details

- **Fifteen categorical hues** (the first eight are the original set), ordered so neighbours stay apart under colour-vision deficiencies. Measured with CIEDE2000 on protan, deutan and tritan simulations, including the wrap from the last slot back to slot 1 at every size from 8 to 15: worst adjacent ΔE is **9.2 light / 10.0 dark** against a floor of 8, and the worst pair for normal vision is **16.2 / 16.0** against a floor of 15. The margin is thinner than with eight hues; `set_style(colors=10)` keeps just the first ten (each size is validated). `tests/unit/test_palette.py` enforces it: **re-order or edit a slot and you must re-validate.**
- **Using the theme in your own traces.** `color(i)` / `palette()` give the categorical colours, `get_style()` the full set of tokens (surface, inks, grid, ramps), and `sequential_scale()` / `diverging_scale()` plotly colourscales for heatmaps and marker colours.
- **Spelling.** The canonical names are American (`color`, `normalize_series`, `normalize_within`, `center=`); `colour`, `normalise_series` and `normalise_within` are exported as aliases.
- **Sequential** is one hue, pale to deep (reversed in dark mode so "more" stays salient). **Diverging** is two poles with a neutral grey midpoint (dark in dark mode). Never a rainbow.
- **Two identity channels that wrap at the same length are one channel.** Colour and shape both have 8 entries, so `marker(i)` shifts the shape on each colour wrap and varies the fill (solid / open / open with dot) after 64: 192 distinct markers. Use `marker()` for scatters and `line_style()` for lines.

## Design rules this library enforces

The library refuses rather than draw something misleading.

- **One y axis, always.** There is no dual-axis option. Two measures of different scale get two charts (`plot_small_multiples`, `plot_grid`), or indexing to a common base.
- **Series caps differ by form, deliberately.** `plot_series`: unlimited, 20 visible. `plot_cumulative`, `plot_drawdown` and `plot_rolling` follow it (`drawn=`). Other line charts (`plot_stacked`, `plot_line_band`, `plot_bars_with_line`, the model-comparison curves): 8 (`MAX_SERIES`). Scatter and facets: 3 (`MAX_ALL_PAIRS`). Coloured by rank or value, and ACF panels: none. `MAX_ALL_PAIRS` is a constant, not `len(palette)`, so growing the palette cannot silently raise the scatter cap.
- **No violin.** A KDE over ~12 points draws density that is not in the data. `plot_box(points=True)` is the honest form below a few hundred observations per group.
- **No pie.** `plot_bars(normalize=True)` shows the same shares readably.
- **Area, not diameter**, carries a bubble's value; **log axes** get readable ticks; **luminance**, not position, decides cell ink.
- Refusals: negative parts in `plot_stacked`; non-numeric `x` in `plot_box_series`; duplicated column labels; an unmatched `group_by_prefix` label; non-Cartesian figures in `plot_views` / `plot_grid`; negative bubble sizes or error bars; a Student-t reference when the fitted ν ≤ 2 (no finite variance to scale to).

## Writing your own plotter

Build the traces, then call `finish()` last:

```python
import plotly.graph_objects as go
from plotters import color, finish, line_style, as_series_dict

def plot_mine(data, title=""):
    fig = go.Figure()
    for i, (name, s) in enumerate(as_series_dict(data).items()):
        dash, width = line_style(i)
        fig.add_trace(go.Scatter(x=s.index, y=s.to_numpy(), name=name,
                                 line={"color": color(i), "dash": dash, "width": width}))
    return finish(fig, title, "subtitle", ylabel="y")
```

`finish()` owns the title, legend placement, grid and fonts, so no plotter sets them itself and they cannot drift apart. It puts the **legend under the plot** (the title is left-aligned and plotly's default corner legend lands on top of it as soon as there is a subtitle), with the offset computed in pixels and converted to plotly's fraction-of-plot-area, so it holds at any figure height. Past 12 legend entries it moves to a scrolling column on the right. Traces with `visible=False` draw no legend entry and are not counted.

**Log axes.** Use `apply_log(fig, "y", low, high)` rather than `type="log"` whenever the data crosses 1: left alone, plotly labels the minor ticks and drops the leading `0.`, so an axis spanning 0.2 to 10 reads `2 3 4 5 6 7 8 9 1 2 3 4`. On plotly 7.1 `range` is in log10 while `tickvals`, trace data and shapes are in data units; **annotations are in log10**. `apply_log` and `plot_series(markers=, bands=, spans=)` handle that.

## Development

```bash
python -m tests.visual            # writes gallery/index.html (open it in a browser)
python -m tests.visual views      # just the cards whose name contains "views" -> gallery/only.html
pytest                            # unit tests + builds every gallery card in both themes
ruff check . && pyright           # lint and type check
```

**The gallery** shows every function in **light and dark side by side**, with the exact call, a short checklist of what you should see, a palette page (swatches, colour-blind simulations, ΔE tables, ramps, markers) and the print figures at true size. Its sections mirror the chart reference above. Add a card by writing a function in `tests/visual/cards_*.py` and decorating it with `@card(section, title, checks=[...])`; the function body is shown on the page, so the example can never drift from what is drawn.

**Type checking.** `pyright` is configured in `pyproject.toml` and the library plus `examples/` check clean with `typeCheckingMode = "basic"`: strict would want an annotation on every parameter, and `data` is deliberately "Series or DataFrame or dict". One pandas-stub fact worth knowing as a caller: `frame[label]` and `frame[mask]` are typed `Series | DataFrame`, so passing them straight into a plotter fails pyright even though it runs. Two idioms avoid it:

```python
ranked = frame.groupby(["a", "b"])["v"].min().reset_index()       # DataFrame (as_index=False is NOT checkable)
subset = pd.DataFrame(frame[frame["a"] == "x"])                   # wrap a boolean mask
```

Inside the library `frames._one_column` does this narrowing once and **raises on a duplicated label**, which used to produce a wrong chart silently.

**Project map.** Flat layout; every public name is re-exported from `plotters/__init__.py`.

```
plotters/
  __init__.py        public names, grouped by theme
  style.py           palette, ramps, color/line_style/marker, plotly template, set_style/use_style
  layout.py          finish(), apply_log(), log_ticks()   (all pixel maths lives here)
  notebook.py  io.py show/show_all/autoshow;  save_html/save_png
  frames.py          input normalisers and reshaping helpers
  stats.py           pure numbers behind the charts (no plotting)
  timeseries.py  analysis.py  diagnostics.py  correlogram.py  forecasting.py
  distributions.py  relationships.py  comparisons.py  categorical.py  matrices.py  learning.py
  views.py           plot_views, annotate_views, plot_grid
  static.py          matplotlib print figures (opt-in)
  _color.py  _checks.py  _errors.py  _types.py   private helpers
tests/unit/          correctness tests;   tests/visual/   the gallery and its cards
examples/            small type-checked usage scripts (run by the tests)
scripts/             install-into.sh: editable install into other venvs
```
