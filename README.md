# plotters

Modular plotly charts for forecasting and model comparison. Generic pandas in, a plotly `Figure` out.

Nothing calls `.show()` — the caller decides. Analysis helpers also return the statistics and series behind their figures.

## What to pass: four input conventions

There are **four**, not one, and picking the wrong one is the most common error. The first argument tells you which:

| Convention | First arg | Who uses it |
|---|---|---|
| **A. Series-like** | `data` — a Series, a **wide** DataFrame (one series per column), or a `{label: Series}` dict | every time-series and distribution plotter, plus `plot_acf` / `plot_acf_grid` |
| **B. Long frame + column names** | `frame` — a **long/tidy** DataFrame, then `x=`, `y=`, `label=` ... as *column name* strings | `plot_tradeoff`, `plot_scatter`, `plot_shaded_scatter`, `plot_rank_heatmap`, `plot_small_multiples`, `plot_box_series` |
| **C. Two positional series** | `bars` (a Series) and `line` (Series or dict) | `plot_bars_with_line` only |
| **D. Wide / square frame as-is** | `frame` — rows and columns are already the two axes, cells are the values | `plot_matrix`, `plot_table_heatmap`, `plot_contours`; `plot_correlation` takes the data, not its matrix |

The shapes are not interchangeable. Verified:

```
plot_series          Series=OK  DataFrame=OK  dict=OK       <- convention A
plot_box             Series=OK  DataFrame=OK  dict=OK
plot_histogram       Series=OK  DataFrame=OK  dict=OK
plot_expectile_curve Series=OK  DataFrame=OK  dict=OK

plot_scatter         Series=AttributeError  DataFrame(wide)=KeyError  dict=AttributeError
plot_tradeoff        Series=AttributeError  DataFrame(wide)=KeyError  dict=AttributeError
plot_scatter         long frame -> OK                                  <- convention B
```

A wide DataFrame given to a convention-B plotter raises `KeyError` (your `x=` is not a column); a Series or dict raises `AttributeError`. Both mean "wrong convention", not "bad data".

### Convention A — Series, wide DataFrame, or dict

`plot_series`, `plot_cumulative`, `plot_drawdown`, `plot_stacked`, `plot_rolling`, `plot_box`, `plot_histogram`, `plot_ecdf`, `plot_qq`, `plot_cumulative_share`, `plot_expectile_curve`.

All three shapes go through `as_series_dict`, so the chart code never knows which it got:

| You pass | Becomes |
|---|---|
| `Series` | one series, named from `.name` |
| `DataFrame` | **one series per column** — so it must be wide, not tidy |
| `{label: Series}` | itself |

Use `long_to_wide(frame, index=, columns=, values=)` to get from tidy to wide, or `group_values(frame, by=, value=)` to get a dict straight from a long frame.

**Subsetting and colour order.** Convention-A plotters take a list that both picks a subset *and* fixes the order, which is what assigns colour slots:

```
series=['c','a']   traces: ['c', 'a']        # your order, not the frame's
default            traces: ['a', 'b', 'c']
series=['zz']      KeyError: "['zz'] not in ['a', 'b', 'c']"
```

The argument is `series=` everywhere **except `plot_box`**, where it is `groups=`. Same behaviour, different name — `plot_box`'s columns are groups rather than series. Two differences there: passing `groups=` **disables the median sort** (your order wins, and `sort=` becomes irrelevant), and the order is read bottom-to-top on the default `horizontal=True`, so the first name you list appears at the top of the chart. It is not about colour — a box summarises many points, so `plot_box` uses a single hue throughout and the category axis does the naming.

### Convention B — long frame plus column names

`plot_tradeoff(frame, x, y, label)`, `plot_scatter(frame, x, y)`, `plot_rank_heatmap(frame, index, columns, values)`, `plot_small_multiples(frame, panel, x, y)`.

Every one of those keyword arguments is a **string naming a column**, never data. One row per observation.

`group=` and `label=` are different things and easy to swap:

```
plot_scatter(long, x='cost', y='error', group='model')  -> traces: ['m1', 'm2']
plot_scatter(long, x='cost', y='error', label='model')  -> 1 trace, point annotations
```

`group=` splits into one coloured series per value (subject to `MAX_ALL_PAIRS = 3`); `label=` keeps one series and writes the text next to each point. `label_from()` builds a label column from several columns.

### The two that look like A but are not

- `plot_bars_with_line(bars, line, ...)` takes two positional arguments. `bars` must be a single Series — a dict raises `AttributeError`. `line` follows convention A, so a dict there compares several predictions against the same bars.
- `plot_paired_difference(data, base=...)` is convention A, but `base=` is the **name of one of the series**, not a column of a long frame. With a dict input, `base` is a dict key.

### `bands=` is not data either

`plot_series(..., bands={"target": 1.0})` — a `{label: y_value}` dict of horizontal reference lines. Scalars, not series.

### `views=` is a dropdown, not a second chart

`plot_series(..., views={"pooled": ["pooled"], "ESc1": ["ESc1 ewrms", "ESc1 har"]})` puts a menu at the top left that swaps between named subsets of the series already passed. Every series is drawn **once** and the menu toggles `visible`, so the data travels with the figure and an exported HTML keeps working with no callback and no server. The first view is shown on open.

A series named in **no** view stays visible in all of them — which is how a pooled baseline sits behind each per-ticker view without being repeated in every list.

`plot_series` has **no total or per-view series limit.** Every eligible series is retained, but only the first 20 start visible. Additional series are available through the legend, including after switching views.

Past `MAX_SERIES` unnamed series, the always-visible rule is dropped and the views are read as an explicit allowlist: a caller showing its best 8 of 42 would otherwise leave 654 columns "unnamed" and therefore drawn in every view.

`drawn=8` starts with eight series and leaves the rest `legendonly` — greyed out in the legend, one click to add. This works with or without views. `drawn=0` (the default) starts with 20; larger values also start with 20. Common baselines precede the view's members and count towards this initial total. Order each view's list best-first; without views, input order (or `series=` order) determines the prefix. You can enable any number afterwards. Large views use a scrollable vertical legend on the right so the plot keeps its height. Colours and dashes eventually repeat. This changes only `plot_series`; other chart forms retain their existing guards. A caller that truncates its data or view lists before calling the plotter still needs its own limit increased.

```python
import plotters as pl

pl.set_style(dark=True)                                          # once, at the top

pl.plot_series(returns, "Daily returns", percent=True)           # one line per column
pl.plot_box(by_group, reference=1.0)                             # rank groups by median
pl.plot_tradeoff(table, x="cost", y="error", label="model", logx=True)
```

## Install

```
pip install -e . --config-settings editable_mode=compat
```

Needs `pandas`, `numpy`, `plotly`. `scipy` only for `plot_qq`, imported lazily.

The `editable_mode=compat` flag matters for autocomplete. A plain `pip install -e .` writes a `.pth` that *executes* an import-hook finder:

```
import __editable___plotters_0_2_0_finder; __editable___plotters_0_2_0_finder.install()
```

Nothing on disk is then named `plotters/`, and Pylance/pyright never run code — so the import resolves at runtime but static analysis reports `Import "plotters" could not be resolved`, with no completion. `compat` writes a plain static path instead, which both follow:

```
$ cat site-packages/__editable__.plotters-0.2.0.pth
/home/user/plotters

$ pyright a_file_importing_plotters.py
0 errors, 0 warnings, 0 informations
```

Redo it in each new env. If you have already installed the other way, reinstalling with the flag is enough; restart the language server afterwards.

## The charts

### Time series — `plotters.timeseries`

All convention **A** (Series / wide DataFrame / dict) except `plot_bars_with_line`.

| Function | Takes | For |
|---|---|---|
| `plot_series` | `data` | one line per series. The workhorse. `percent`, `logy`, `logx`, `slider`, `fill`, `bands` (horizontal lines), `markers` (vertical), `views` (dropdown over named subsets), `drawn` (start with at most 20; unlimited extras selectable in the legend) |
| `plot_bars_with_line` | `bars` (Series), `line` (A) | realised as bars, prediction as a line, one axis. Did it track or lag? |
| `plot_cumulative` | `data` | wealth curve from per-period values. `compound=True` for reinvested |
| `plot_drawdown` | `data` | drop from the running peak, filled because the area *is* the quantity |
| `plot_stacked` | `data` | parts of a whole. Refuses negative parts |
| `plot_rolling` | `data`, `window` (int) | a rolling statistic over the raw series in lighter ink. `how="mean"`/`"std"`/… |

All of these take `series=` to subset and order, except `plot_bars_with_line`.

### Series analysis — `plotters.analysis` / `plotters.statistics`

Convention A. Use the individual functions for one table or chart, or `analyze_series` for the table and the paths together.

| Function | Takes | For |
|---|---|---|
| `analyze_series` | `series` | statistics and five figures: values, absolute values, cumulative sums, cumulative absolute sums, histogram. `normalize=`, `center=`, `pooled=`, `bins=`, `quantiles=` |
| `compare_series` | `data`, `reference=` | several series on the same dates, their cumulative paths and differences against one reference. `align="inner"`/`"outer"` |
| `plot_series_distribution` | `data` | histogram with a dropdown, statistical annotations and quantile markers. `statistics=`, `display_stats=`, `quantiles=`, `normal_reference=` |
| `series_statistics` | `data` | one row per series: moments, quantiles, tail frequencies and turnover. `thresholds=` changes the tail cut-offs; `metrics={name: callable}` adds scalar statistics |
| `series_turnover_ratio` | `series` | mean absolute adjacent change / mean absolute level. `group_level=` keeps MultiIndex differences within each series |
| `normalise_series` | `data`, `method=` | transformed series, means and scales. `method="std"`/`"rms"`/`None`, `center=`, `ddof=` |

```python
analysis = pl.analyze_series(
    data, bins=100, pooled=True,
    normalize=None, center=False,
    quantiles=(0.01, 0.5, 0.99),
    display_stats=("mean", "std", "skewness", "excess_kurtosis", "turnover_ratio"),
)
display(analysis["stats"])
for fig in analysis["figures"].values():
    display(fig)
```

**A pooled distribution and a pooled path are different quantities.** The histogram stacks observations with equal weight per observation; the time plots average available series at each index value. Absolute paths average magnitudes *before* accumulation. Set `pooled=False` when combining the inputs' units would make no sense. Each series still has its own dropdown view.

`normalize="std"` or `"rms"` scales each series by its own full retained sample; `center=True` also subtracts its mean. This is descriptive normalisation, not a causal transform. `cut` removes leading rows after sorting each input's index. Reference series must already be in the plotted units; they are not rescaled automatically.

**Histogram limits do not erase tail probability.** Values outside `hist_range` stay in the density denominator. Quantile markers use the full sample and change with the dropdown. `normal_reference=True` overlays N(0,1), not a fitted normal. Tail frequencies in the table refer to the input values; no standardisation is implied.

Missing and infinite values are excluded from moments. They break adjacent turnover differences rather than creating a change across the gap. Default pooled turnover keeps differences within each input series; a custom `turnover_ratio` callback is pooled using observation-count weights. Standard deviation uses `ddof=0`; skewness and excess kurtosis use pandas' sample corrections.

`compare_series(data, reference="baseline")` returns `data`, `cumulative`, `difference`, `stats` and `figures`. Differences are *reference minus each other series*. `align="inner"` uses common finite observations for all inputs; `"outer"` retains gaps and each difference uses its own available pairs. More series do not create all pairs. The first series is the reference when none is named.

### Figure views — `plotters.views`

Prepared figures in, one dropdown figure out. Use this when the views have different annotations, shapes or axes; use `plot_series(views=)` for subsets of the same lines.

| Function | Takes | For |
|---|---|---|
| `plot_views` | `{view_name: Figure}` | switch traces, axes, legend, annotations and shapes together. `title=`, `height=`, `dropdown_x=`, `dropdown_y=` |
| `annotate_views` | `fig`, `{view_name: text}` | add a changing top-right annotation to an existing dropdown; modifies and returns the figure |

```python
fig = pl.plot_views({"Raw": raw_figure, "Normalised": normalised_figure})
```

The first view opens initially. Source figures are copied, including their originally hidden traces. Cartesian figures only: existing menus, sliders, animation frames and non-Cartesian traces raise rather than losing their interactions.

### Distributions — `plotters.distributions`

All convention **A**. Each series is treated as a bag of values — the index is ignored, so these take an unindexed Series happily.

| Function | Takes | For |
|---|---|---|
| `plot_null_test` | `null`, `observed`, `alpha=` | a simulated distribution with the tested value on it and the rejection region shaded — **the picture of a hypothesis test**. Reads either way round: null centred on H0 with `observed` = the statistic, or the statistic's own bootstrap with `observed` = 0. The p is counted from the distribution, not read off a normal |
| `plot_box` | `data`, `groups=` | one box per group, sorted by median. **The ranking chart.** `reference=`, `points=`, `sort=`, `name_points=`, `logvalue=` (log value axis, for a tail decades past the box) |
| `plot_histogram` | `data`, `series=` | overlaid distributions, optional normal fit. `bins=`, `density=` |
| `plot_ecdf` | `data`, `series=` | the distribution with no bin-width choice. Better for tails |
| `plot_qq` | `data`, `series=`, `dist=` | QQ against the normal, or `dist="t"` for Student-t with `nu` fitted by MLE. Reference line fitted through the IQR |
| `plot_cumulative_share` | `data`, `series=` | how concentrated a total is, against a "paid evenly" diagonal |
| `plot_expectile_curve` | `data`, `series=`, `taus=` | the tau expectile over each series' OWN mean, against tau. **The slope is the reading** |

Note `plot_box` spells the subset argument `groups=`; the other five use `series=`.

`plot_expectile_curve` answers "do a few large observations carry the mean?" Every curve is divided by its own mean, so all of them pass through 1 at `tau = 0.5` and the only thing to read is the slope of the right half. Unlike a quantile, an expectile responds to how *large* the exceedances are and not only how many. Measured on two lognormals, `expectile(0.95) / mean`: 1.75 at sigma 0.5 against 5.01 at sigma 1.5. Normalising per series is deliberate — the question is concentration, not level, so series on different scales belong on one picture. `sample_expectile(values, tau)` is exported from the same module if you want the number rather than the chart.

`plot_qq(dist="t")` is the one to reach for on fat tails. A normal QQ on a fat-tailed series bends at both ends and stops there — it says the normal is wrong, not what is right. The t reference answers that, and `nu` comes from an MLE fit per series unless you pass it. On a simulated `t(5)` sample the departure from the reference line falls from 6.169 to 1.472 and the fit recovers `nu = 5.47`. `standardise` defaults ON for `dist="t"`, since the t quantiles are scaled to unit variance.

### Autocorrelation — `plotters.correlogram`

Convention A. The form is deliberately statsmodels' `plot_acf`: a stem from zero to each coefficient, a marker at the tip, the significance band as a ribbon behind.

| Function | Takes | For |
|---|---|---|
| `plot_acf` | `data`, `lags`, `absolute=` | one correlogram, band drawn when there is a single series |
| `plot_acf_grid` | `data`, `lags`, `ncols=` | one panel per series, **each with its own band**. No series cap |
| `acf_values` | `series`, `lags` | the coefficients alone, denominator `N` |
| `bartlett_band` | `coefficients`, `n`, `alpha=` | the half-width at each lag |

**The band widens with the lag, and that is the point.** Bartlett's formula is `var(r_k) = (1 + 2·Σ_{j<k} r_j²)/n`, so the sampling error at lag *k* inherits the autocorrelation already present at shorter lags; it collapses to the textbook `1.96/√n` only at `k = 1`. A flat line is wrong at every other lag. Checked against statsmodels: coefficients agree to 4e-16 and the band to 1.2e-06 (the residual is a rounded z). No statsmodels dependency.

`absolute=True` plots the acf of `|x|`. On a return series the two ask different questions — `acf(x)` tests whether the **sign** is predictable, `acf(|x|)` whether the **magnitude** is, which is volatility clustering. A series can be serially uncorrelated and still strongly clustered; that is the ARCH case. The band is pointwise, so ~1 lag in 20 escapes by chance: read the profile, not one stem.

`plot_acf_grid` exists rather than overlaying because the band depends on the series' own length *and* its own shorter-lag coefficients, so a shared band would be wrong for all but one series.

### Matrices — `plotters.matrices`

Convention D. These colour by **value**, where `plot_rank_heatmap` colours by rank.

| Function | Takes | For |
|---|---|---|
| `plot_matrix` | `frame`, `diverging=`, `centre=`, `mask_diagonal=` | any wide/square frame on its own values |
| `plot_correlation` | `frame` (the *data*), `method=` | correlation between columns, diverging about 0, diagonal masked |
| `plot_table_heatmap` | `frame`, `normalise="column"` | a table whose columns are on **different scales** — a moments table |
| `plot_contours` | `frame`, `logx=`, `logy=`, `diagonal=` | a function of two variables as labelled contours. Index is y, columns are x, cells the value |

`plot_contours` takes the grid already evaluated, so the caller owns the maths: build `frame` with `np.meshgrid` and whatever function, then pass it. Contours rather than a 3-D surface on purpose — a surface hides its own far side and cannot be read off, while labelled level curves give exact values. `diagonal=True` draws `y = x`, the line where the two arguments agree, which is the reference for any `f(a, b)` that should vanish at `a = b`.

**When rank is the wrong encoding.** A rank discards magnitude, which is fine for "which model is best in this column" and wrong everywhere else. Measured on a correlation matrix: `corr = 0.971` and `corr = -0.020` both come out rank 2, i.e. the same colour, and the subtitle claims "1 = best". A moments table is worse — there is no "better", ranking `skew` ascending calls `-1.2` the worst when its magnitude makes it the most notable, and a constant column like `n` gets a meaningless uniform rank.

`plot_matrix` picks the scale from the data: a frame straddling `centre` gets the two diverging poles with the range forced symmetric, or the neutral midpoint would lie about where zero is. One that does not straddle gets the single-hue ramp.

`plot_correlation` takes the **data frame, not a correlation matrix** — computing it inside is what keeps the scale symmetric and the diagonal masked. The diagonal is 1.0 by construction and would otherwise own the whole top of the scale. Pass `method="spearman"` on fat-tailed series.

The ink on each cell is chosen from the fill's **measured luminance**, not from a position threshold: the dark-mode diverging ramp has a dark midpoint, so a threshold rule put dark text on dark fill for 9 cells of 21. Worst fill-to-ink luminance gap is now 0.450 across both modes and both scales, against a floor of 0.33.

### Print figures — `plotters.static`

Not imported by `plotters/__init__.py` — it needs matplotlib, which the plotly side does not. `from plotters import static as ps`.

```python
import plotters as pl
from plotters import static as ps

pl.set_style(dark=False)        # same palette as the plotly charts
ps.use_print_style()            # rcParams : serif, 9pt, the palette's inks
figure = ps.plot_lines(frame, title="...", ylabel="...", label_last=True)
ps.save(figure, "figures/thing.png")    # 300dpi, closes the figure
```

| Function | Takes | For |
|---|---|---|
| `plot_lines` | `data` (A) | one line per series. `label_last=` writes names at the line ends instead of a legend |
| `plot_points` | `frame`, `x`, `y` (B) | scatter, optional `frontier=True` for the Pareto front |
| `plot_ranked_bars` | `values` (Series) | sorted bars with the value written at each tip. **The ranking figure for print** |
| `plot_heatmap` | `frame` (D) | a matrix on its own values, every cell written in. `diverging=`, `centre=` |
| `plot_function` | `{label: callable}`, `x` | the shape of a loss, not data. `marks=` for a vertical reference |
| `plot_stems` | `values`, `band=` | a correlogram. `band` is the half-width per lag, so `bartlett_band`'s widening draws itself |
| `plot_panels` | `{panel: data}` | a grid, one shared legend, labels on the edges only |

`figure_size(width=0.86)` gives inches for that fraction of `\textwidth` at a4paper with 2.2cm margins. **Size here, never with `includegraphics[scale=]`** — scaling in LaTeX scales the figure's text too, so two figures at different scales end up with different apparent font sizes on the same page.

**What changes for print, and why.**

- `vary_dash` defaults ON, unlike `line_style`'s first six solid slots. On screen the legend isolates a series on click; a printed page has no legend to click, and photocopied or read by a colourblind reader four solid lines are one line.
- End-of-line labels are nudged apart (`_label_ends`). Two curves finishing at the same height overprinted and both became unreadable — worse than the legend it replaces.
- Subtitles wrap to the axes width. A figure is embedded at its own size, so an overlong subtitle runs off the *page*; nothing clips it.
- Cell ink comes from measured luminance, as in `matrices` — the diverging ramp's midpoint is dark in one mode and light in the other.
- `percent=True` writes a bare `%`. The PNG is a raster, so matplotlib draws that text and a LaTeX escape would print the backslash.

### Comparisons — `plotters.comparisons`

Convention **B** (long frame + column-name strings), except `plot_paired_difference` which is convention A.

| Function | Takes | For |
|---|---|---|
| `plot_tradeoff` | `frame`, `x`, `y`, `label`, `group=`, `groups=` | two competing objectives + the Pareto frontier. "You cannot have both". `lower_is_better=(bool, bool)` says which corner wins. `groups={family: prefixes}` collapses `group`'s values by longest prefix, so colour means family and shape means member. `connect=` joins each group's points along a numeric column, turning a cloud into one path per group |
| `plot_rank_heatmap` | `frame`, `index`, `columns`, `values` | items × contexts coloured by rank. No series cap. `index` = rows, `columns` = contexts |
| `plot_box_series` | `frame`, `x=`, `y=`, `group=`, `label=`, `points=` | a box at each numeric `x`, medians joined so each group reads as a curve. The line is the subject, the boxes are its error bars |
| `plot_small_multiples` | `frame`, `panel`, `x`, `y`, `group=` | one panel per value of `panel`, shared legend. `ncols=`, `shared_y=` |
| `plot_paired_difference` | `data` (A), `base` | cumulative difference vs a baseline. Steady slope or one cliff? `base` names a series, not a column |
| `plot_scatter` | `frame`, `x`, `y`, `group=` / `label=`, `xlabel=` / `ylabel=` | two columns, optional `y = x` diagonal |
| `plot_shaded_scatter` | `frame`, `x`, `y`, `shade`, `label=`, `log_shade=`, `reference=` | two columns plus a **continuous** third as colour, with a colourbar. Use instead of `plot_scatter(group=)` when the third variable is a quantity rather than a category — `group=` caps at three hues and throws the gradient away. `log_shade=True` for a geometric ratio |
| `plot_interval` | `frame`, `low`, `high`, `point`, `label=`, `group=`, `reference=` | one horizontal interval per row with a marker at `point` — "is this difference bigger than its own uncertainty". An interval crossing `reference` is the visual form of *not separable*. `sort=` orders by `point`, `lower_is_better=` is a plain `bool` |

`plot_box_series` needs a **NUMERIC x, and that is not a style choice.** A plotly `Box` width is in data units, so the same `width` renders many times wider at one end of a log axis than the other — which is why the x axis here is linear, ticked at the values that exist, and a non-numeric `x` raises. The width itself comes from the *tightest* gap between two x values divided by the group count, so the cluster occupies a fixed share of it whatever the series count; a hardcoded width fills most of the narrowest gap on an uneven grid. And the boxes sit at an *explicit* x with a per-group nudge rather than using `boxmode="group"`, because a grouped box is offset from its category centre and the median line would then miss its own boxes.

`x`, `y`, `label`, `group`, `panel`, `index`, `columns`, `values` are all **column-name strings**. `lower_is_better` is the one that bites: it is a 2-tuple on `plot_tradeoff` (one flag per axis) but a plain bool on `plot_rank_heatmap`.

### Data shaping — `plotters.frames`

Useful on their own, and the bridge between the two conventions — all take a **long frame** and a few column names:

| Function | Takes | Returns |
|---|---|---|
| `long_to_wide` | `frame`, `index`, `columns`, `values` | wide DataFrame, for convention A. **Reorders numeric columns numerically** — otherwise `pivot` sorts them as strings and `126` lands between `18` and `21`, which makes every line chart zigzag |
| `group_values` | `frame`, `by`, `value`, `keep=` | `{group: Series}`, for the distribution plotters. Ordered by median unless `keep` fixes it |
| `normalise_within` | `frame`, `value`, `within`, `how="min"` | Series — `value` divided by its own group's reference. The fix for pooling across groups at different levels |
| `group_by_prefix` | `labels`, `{group: (prefix, ...)}`, `other=` | Series of group names, for `group=`. Matches the **longest** prefix, so `ewrms_gk` lands in `range` rather than `ewrms` and the dict's order cannot decide it. Raises on an unmatched label unless `other` is given — a silent bucket hides the entry you forgot |
| `label_from` | `frame`, `*columns`, `sep=" "` | Series of labels, for `label=`. Skips NaN parts, so a row with no parameter reads `model` not `model nan` |
| `as_series_dict` | `data`, `value=` | `{label: Series}` — the convention-A normaliser itself |
| `pick` | `series` (dict), `keep=` | subset **in the order given**, which is what `series=` uses internally |

## Type checking

`pyright` is configured in `pyproject.toml` and the library plus `examples/` check clean. A library other projects import has to be, or every caller inherits the noise. `typeCheckingMode = "basic"` rather than `strict`: strict wants an annotation on every parameter, and `data` is deliberately "Series or DataFrame or dict".

**The one pandas-stub fact worth knowing as a caller.** `frame[label]` and `frame[mask]` are typed `Series | DataFrame`, so passing either straight into a plotter fails pyright even though it runs. Two idioms avoid it:

```python
# groupby : .reset_index() type-checks, as_index=False does NOT
ranked = frame.groupby(["a", "b"])["v"].min().reset_index()      # DataFrame
ranked = frame.groupby(["a", "b"], as_index=False)["v"].min()    # NDFrame | Series

# a boolean mask : wrap it
subset = pd.DataFrame(frame[frame["a"] == "x"])
```

Both forms return a DataFrame at runtime; only the first is checkable. Inside the library, `frames._one_column` does this narrowing once and **raises on a duplicated label** — which used to produce a wrong chart silently.

## Design rules this library enforces

These are the ones worth knowing, because the library will refuse rather than draw something misleading.

**One y axis, always.** There is no dual-axis option. Two measures of different scale get two charts (`plot_small_multiples`), or indexing to a common base — a second axis lets you imply any correlation you like by sliding one scale.

**Series caps differ by chart form, and that is deliberate.**

| Form | Cap | Why |
|---|---|---|
| `plot_series` | unlimited; at most 20 initially visible | extra series stay selectable in the legend; colours/styles eventually repeat |
| Other line charts | existing `MAX_SERIES` colour guard | unchanged by the `plot_series` exception |
| Scatter, facets | 3 (`MAX_ALL_PAIRS`) | read *pair by pair*; no ordering of 8 hues clears the all-pairs floors past 3 |
| Coloured by rank or value | none | colour encodes magnitude, not identity |
| ACF panels | none | one series per panel, so colour names nothing |

`MAX_ALL_PAIRS` is a constant rather than `len(palette)`, so growing the palette cannot silently raise the scatter cap.

**Colour comes last, and the order is the safety mechanism.** The eight categorical hues were validated with a colourblind-separation checker in both modes — worst adjacent pair ΔE 9.1 light / 8.4 dark against a floor of 8, worst normal-vision pair 19.6 / 19.3 against a floor of 15. **Re-order the slots and you must re-validate.**

**Two identity channels that wrap at the same length are one channel.** Shape and colour both have 8 entries, so cycling them together makes slot 8 identical to slot 0 in *both*. `marker()` varies the fill on each wrap (solid / open / open-dot) for 8 × 8 × 3 distinct markers. Use `marker()` for scatters, `line_style()` for lines.

**No violin option.** A KDE over ~12 points draws density that is not in the data: it spills past hard floors and invents modes from single observations. `plot_box` with `points=True` is the honest form below a few hundred observations per group.

**Sequential = one hue light→dark. Diverging = two poles + a neutral grey midpoint. Never a rainbow.**

## Writing your own plotter

Build the traces, then call `finish()` last:

```python
import plotly.graph_objects as go
from plotters import colour, finish, line_style, as_series_dict

def plot_mine(data, title=""):
    frames = as_series_dict(data)
    fig = go.Figure()
    for i, (name, s) in enumerate(frames.items()):
        dash, _ = line_style(i)
        fig.add_trace(go.Scatter(x=s.index, y=s.to_numpy(), name=name,
                                 line={"color": colour(i), "dash": dash}))
    return finish(fig, title, "subtitle", ylabel="y")
```

`finish()` owns the title, legend placement, grid and fonts, so no plotter sets them itself and they cannot drift apart. It puts the **legend under the plot**, because the title is left-aligned and plotly's default corner legend lands on top of it the moment there is a subtitle; the offset is computed in pixels and converted to plotly's fraction-of-plot-area units, so it holds at any figure height.

Use `apply_log(fig, "x", low, high)` rather than `type="log"` whenever the data crosses 1 — left alone, plotly labels minor log ticks and drops the leading `0.`, so an axis spanning 0.2 to 10 reads `2 3 4 5 6 7 8 9 1 2 3 4`.

**On a log axis every axis-space coordinate is in log10** — `tickvals`, `range`, and a shape's `x`. Only `ticktext` and the trace data stay in data units. A `tickval` of 3 is placed at 10^3, and with autorange the axis then stretches three decades to reach it, crushing the real data into a sliver at one edge. `apply_log` converts the ticks and sets an explicit range; `plot_series` does the same for `markers=`. A static `write_image` hides the bug, because kaleido ranges off the trace data.

## Notebook gotcha

These return a Figure and draw nothing, so Jupyter renders only the **last expression in a cell**. Two calls in one cell shows one chart. Loop with `display`:

```python
from IPython.display import display
for column in frame.columns:
    display(pl.plot_series(frame[column], column))
```

## Saving

```python
fig.write_html("chart.html")                          # keeps interactivity
fig.write_html("chart.html", include_plotlyjs="cdn")  # ~3MB smaller, needs internet
```

`write_image("x.png")` needs kaleido **and** a Chrome install; on a headless box it usually fails. If you need a PNG and cannot get one, mirror the figure in matplotlib at the same height to eyeball the layout.
