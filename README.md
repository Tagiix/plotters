# plotters

Generic plotly charts with one consistent, colour-blind-safe look. Pandas in, a themed plotly `Figure` out.

Nothing calls `.show()`: the caller decides. Analysis helpers also return the statistics and series behind their figures.

```python
import plotters as pl

pl.set_style(dark=False)                                  # once, at the top

pl.plot_series(returns, "Daily returns", percent=True)    # one line per column
pl.plot_box(by_group, reference=1.0)                      # rank groups by median
pl.plot_bars({"cats": 42, "dogs": 31, "birds": 12})       # straight from a dict of counts
pl.plot_tradeoff(table, x="cost", y="error", label="model", logx=True)
```

## Install

```bash
pip install "plotters @ git+https://github.com/<you>/plotters.git"             # core
pip install "plotters[stats,static] @ git+https://github.com/<you>/plotters.git"  # + scipy, matplotlib
uv add "plotters @ git+https://github.com/<you>/plotters.git@v0.1.0"          # pin a tag
```

Core needs `pandas`, `numpy`, `plotly`. `scipy` is only for `plot_qq(dist="t")` (imported lazily) and `matplotlib` only for `plotters.static`.

For development (editable, with autocomplete working in VS Code / Pylance):

```bash
git clone https://github.com/<you>/plotters && cd plotters
uv venv && uv pip install -e ".[dev]"
```

The build backend (hatchling) writes a plain static path for editable installs, so Pylance and pyright resolve the import with no extra flag.

## Hover documentation

Every public function has a docstring listing its arguments, defaults and what it raises, and `Literal[...]` types show the allowed values. In VS Code, hover a function or press `K` on it to read it.

## What to pass: five input conventions

The first argument tells you which one a plotter uses. Picking the wrong one is the most common error, so the library names the convention in a `ConventionError` instead of failing deep inside pandas.

| Convention | First argument | Used by |
|---|---|---|
| **A. Series-like** | `data`: a Series, a **wide** DataFrame (one series per column), or `{label: Series}` | time series, distributions, correlogram, analysis |
| **B. Long frame + column names** | `frame`: a **long/tidy** DataFrame, then `x=`, `y=`, `label=` ... as *column name* strings | comparisons |
| **C. Two positional series** | `bars` (a Series) and `line` (Series or dict) | `plot_bars_with_line` |
| **D. Wide / square frame as-is** | `frame`: rows and columns are the two axes, cells are the values | matrices, `plot_grouped_bars` |
| **E. Label → number** | a `{label: number}` dict, or a Series indexed by label (`value_counts()` works) | `plot_bars`, `plot_lollipop`, `plot_waterfall` |

```python
pl.plot_scatter(frame_wide)            # ConventionError: plot_scatter expects a long (tidy) DataFrame
                                       # followed by column-name strings (convention B); got dict.
```

Convert between shapes with `long_to_wide`, `group_values`, `as_series_dict`, `as_value_series`.

**Subsetting and colour order.** Convention-A plotters take `series=[...]`, which both picks a subset *and* fixes the order, and the order assigns the colour slots. An unknown name raises `KeyError: "['zz'] not in ['a', 'b', 'c']"`. It is `series=` everywhere, including `plot_box`.

**`group=` and `label=` differ.** `group="model"` splits into one coloured series per value (at most 3, since scatter is read pair by pair); `label="model"` keeps one series and writes the text next to each point. `groups={family: prefixes}` collapses many members into families (the longest prefix wins) so colour means family and shape means member.

## The charts

### Time series: `timeseries`

| Function | For |
|---|---|
| `plot_series` | One line per series: the workhorse. `percent`, `logy`, `logx`, `slider`, `fill`, `bands` (horizontal reference lines), `markers` (vertical), `views` (dropdown over named subsets), `max_visible` |
| `plot_bars_with_line` | Realised as bars, prediction as lines on one axis. Did it track or lag? |
| `plot_cumulative` | Wealth curve from per-period values; `compound=True` for reinvested |
| `plot_drawdown` | Drop from the running peak, filled because the area *is* the quantity |
| `plot_stacked` | Parts of a whole; refuses negative parts; `normalize=True` for shares |
| `plot_rolling` | A rolling statistic over the raw series in lighter ink |

`plot_series` has **no series limit**: every series is kept but only the first `max_visible` (default 20) start drawn, the rest sit greyed out in the legend, one click to add. With `views={"A": [...], "B": [...]}` a dropdown swaps named subsets of the lines already passed (each series is drawn once and the menu toggles `visible`, so an exported HTML keeps working with no server). A series named in no view stays visible in all of them: a pooled baseline sits behind each per-ticker view without repeating it. Past 8 such series the views become an explicit allowlist.

### Statistics and analysis: `stats`, `analysis`

| Function | For |
|---|---|
| `analyze_series` | Statistics plus five figures (values, absolute values, cumulative sum, cumulative absolute sum, histogram), each with a pooled view and one view per series |
| `compare_series` | Several series on the same dates against one reference (`align="inner"/"outer"`); differences are reference minus the others |
| `plot_series_distribution` | Histogram with a dropdown, statistics block and quantile markers |
| `series_statistics` | One row per series: moments, quantiles, tail frequencies, turnover; `metrics={name: callable}` adds columns |
| `series_turnover_ratio` | Mean absolute adjacent change over mean absolute level; gaps break the differences |
| `normalize_series` | Per-series scaling (`"std"` / `"rms"`), returns the series, means and scales |
| `sample_expectile`, `acf_values`, `bartlett_band` | The numbers behind `plot_expectile_curve` and the correlogram |

```python
out = pl.analyze_series(returns, bins=100, normalize="std", quantiles=(0.01, 0.5, 0.99))
out["stats"]                          # DataFrame
pl.show_all(out["figures"])           # every figure from one notebook cell
```

A pooled **distribution** and a pooled **path** are different quantities: the histogram stacks observations with equal weight per observation, while the time plots average the series available at each index value. Set `pooled=False` when combining the inputs' units makes no sense.

Histogram limits (`hist_range`) never erase tail probability: values outside the window stay in the density denominator. `normal_reference=True` overlays N(0, 1), not a fitted normal.

### Figure views: `views`

`plot_views({"Raw": fig_a, "Normalised": fig_b})` merges prepared figures into one with a dropdown; traces, axes, annotations, shapes and legend switch together. `annotate_views(fig, {view: text})` adds a top-right annotation that changes with the dropdown. Cartesian figures only; existing menus, sliders, frames and non-Cartesian traces raise rather than silently losing interactions.

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

### Autocorrelation: `correlogram`

`plot_acf` draws a stem from zero to each coefficient, a marker at the tip, and the significance band as a ribbon. **The band widens with the lag**, and that is the point: Bartlett's formula inherits the autocorrelation already present at shorter lags and equals the textbook `1.96/√n` only at lag 1. `absolute=True` plots `acf(|x|)`: on a return series `acf(x)` asks whether the sign is predictable, `acf(|x|)` whether the magnitude is (volatility clustering, the ARCH case). `plot_acf_grid` gives each series its own panel and band, with no series cap.

### Matrices: `matrices`

Colour by **value** (a rank would discard magnitude).

| Function | For |
|---|---|
| `plot_matrix` | Any wide/square frame. A frame straddling `center` gets the diverging ramp with a symmetric range, otherwise the single-hue ramp |
| `plot_correlation` | Correlation between columns (takes the **data**), diverging about 0, diagonal masked; `method="spearman"` for fat tails |
| `plot_table_heatmap` | A table whose columns are on different scales (a moments table); each column shaded on its own range, actual values written in |
| `plot_contours` | A function of two variables as labelled level curves, from a grid you evaluated yourself |

Cell text is coloured from the **measured luminance** of each fill, never from a position threshold: the dark-mode diverging ramp has a dark midpoint, so a threshold rule puts dark text on dark fill.

### Comparisons: `comparisons`

| Function | For |
|---|---|
| `plot_tradeoff` | Two competing objectives and the Pareto frontier; `lower_is_better`, `groups=`, `connect=` |
| `plot_rank_heatmap` | Items by contexts coloured by rank within each context (the one chart that colours by rank) |
| `plot_box_series` | A box at each numeric `x`, medians joined into a curve per group |
| `plot_small_multiples` | One panel per value of `panel`, shared legend |
| `plot_paired_difference` | Cumulative difference against a baseline series (`base` names a series) |
| `plot_scatter` | Two columns, optional `y = x` diagonal |
| `plot_shaded_scatter` | Two columns plus a continuous third as colour, with a colourbar |
| `plot_interval` | One horizontal interval per row; intervals crossing `reference` are hollow (not separable) |

`plot_box_series` needs a **numeric x**, and that is not a style choice: a plotly `Box` width is in data units, so the same width renders many times wider at one end of a log axis than the other. The axis is linear, ticked at the values that exist, and the width comes from the tightest gap divided by the group count.

`lower_is_better` is a bool or a `(x_flag, y_flag)` pair wherever two axes exist.

### Categorical and relationships: `categorical`

| Function | For |
|---|---|
| `plot_bars` | Bars from a `{label: number}` dict or a Series, value written at each tip. `sort`, `top`, `normalize` |
| `plot_counts` | `value_counts` and the bars in one call |
| `plot_grouped_bars` | A few series across categories; `stacked`, `normalize` |
| `plot_lollipop` | Stems and dots: less ink, good for many categories or negative values |
| `plot_dumbbell` | Before and after per item |
| `plot_waterfall` | Contributions adding up to a total |
| `plot_line_band` | A line with an uncertainty ribbon |
| `plot_regression` | Scatter with an OLS fit and a 95% band (numpy only) |
| `plot_scatter_matrix` | Pairwise scatter of up to 6 columns with diagonal histograms |
| `plot_calendar_heatmap` | A daily series as a weekday-by-week grid |

### Data shaping: `frames`

| Function | Returns |
|---|---|
| `long_to_wide(frame, index, columns, values)` | Wide DataFrame. **Orders numeric columns numerically**, otherwise `pivot` sorts them as strings and `126` lands between `18` and `21` |
| `group_values(frame, by, value, keep=)` | `{group: Series}`, ordered by median unless `keep` fixes it |
| `normalize_within(frame, value, within, how="min")` | `value` divided by its own group's reference |
| `group_by_prefix(labels, {group: (prefix, ...)}, other=)` | Group names by the **longest** matching prefix; raises on an unmatched label unless `other` is given |
| `label_from(frame, *columns, sep=" ")` | Labels for `label=`; NaN parts are skipped |
| `as_series_dict`, `as_value_series`, `pick` | The convention normalisers, and subsetting in the order given |

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

## Theme

`set_style(dark=False)` chooses the palette and installs a plotly template, so even a plain `go.Figure()` created afterwards gets the fonts, colours and grid. `use_style(dark=True)` is a context manager that restores the previous theme. `set_style("auto")` reads the `PLOTTERS_STYLE` environment variable.

- **Eight categorical hues**, ordered so neighbours stay apart under colour-vision deficiencies. Measured with CIEDE2000 on protan, deutan and tritan simulations (including the wrap from slot 8 back to slot 1): worst adjacent ΔE is **15.6 light / 17.3 dark** against a floor of 8, and the worst pair for normal vision is **26.3** against a floor of 15. `tests/unit/test_palette.py` enforces it: **re-order or edit a slot and you must re-validate.**
- **Sequential** is one hue, pale to deep (reversed in dark mode so "more" stays salient). **Diverging** is two poles with a neutral grey midpoint (dark in dark mode). Never a rainbow.
- **Two identity channels that wrap at the same length are one channel.** Colour and shape both have 8 entries, so `marker(i)` shifts the shape on each colour wrap and varies the fill (solid / open / open with dot) after 64: 192 distinct markers. Use `marker()` for scatters and `line_style()` for lines.

## Design rules this library enforces

The library refuses rather than draw something misleading.

- **One y axis, always.** There is no dual-axis option. Two measures of different scale get two charts (`plot_small_multiples`), or indexing to a common base.
- **Series caps differ by form, deliberately.** `plot_series`: unlimited, 20 visible. Other line charts: 8 (`MAX_SERIES`, one per colour). Scatter and facets: 3 (`MAX_ALL_PAIRS`). Coloured by rank or value, and ACF panels: none. `MAX_ALL_PAIRS` is a constant, not `len(palette)`, so growing the palette cannot silently raise the scatter cap.
- **No violin.** A KDE over ~12 points draws density that is not in the data. `plot_box(points=True)` is the honest form below a few hundred observations per group.
- **No pie.** `plot_bars(normalize=True)` shows the same shares readably.
- Refusals: negative parts in `plot_stacked`; non-numeric `x` in `plot_box_series`; duplicated column labels; an unmatched `group_by_prefix` label; non-Cartesian figures in `plot_views`; a Student-t reference when the fitted ν ≤ 2 (no finite variance to scale to).

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

**Log axes.** Use `apply_log(fig, "y", low, high)` rather than `type="log"` whenever the data crosses 1: left alone, plotly labels the minor ticks and drops the leading `0.`, so an axis spanning 0.2 to 10 reads `2 3 4 5 6 7 8 9 1 2 3 4`. On plotly 7.1 `range` is in log10 while `tickvals`, trace data and shapes are in data units; **annotations are in log10**. `apply_log` and `plot_series(markers=, bands=)` handle that.

## Notebooks

These functions return a Figure and draw nothing, so Jupyter renders only the **last expression in a cell**. Three ways around it:

```python
pl.show(fig_a, fig_b)                 # explicit, any number
pl.show_all(out["figures"])           # a dict or list of figures
pl.autoshow(True)                     # every plotter displays as soon as it finishes, mid-cell
```

`autoshow` hooks `finish()`, so it also works for your own plotters. Assign the result (`fig = pl.plot_series(...)`) or end the line with `;` to avoid Jupyter showing it twice.

## Saving

```python
pl.save_html(fig, "chart.html")                 # keeps interactivity, self-contained
pl.save_html(fig, "chart.html", cdn=True)       # ~3 MB smaller, needs internet
pl.save_png(fig, "chart.png")                   # needs kaleido and Chrome (plotly_get_chrome)
```

`write_image` needs kaleido **and** a Chrome install; on a headless box it usually fails. For PNGs without Chrome use `plotters.static`.

## Validating the visuals

```bash
python -m tests.visual            # writes gallery/index.html (open it in a browser)
python -m tests.visual views      # just the cards whose name contains "views" -> gallery/only.html
pytest                            # unit tests + builds every gallery card in both themes
```

The gallery shows every function in **light and dark side by side**, with the exact call, a short checklist of what you should see, a palette page (swatches, colour-blind simulations, ΔE tables, ramps, markers) and the print figures at true size. Add a card by writing a function in `tests/visual/cards_*.py` and decorating it with `@card(section, title, checks=[...])`; the function body is shown on the page, so the example can never drift from what is drawn.

## Type checking

`pyright` is configured in `pyproject.toml` and the library plus `examples/` check clean with `typeCheckingMode = "basic"`: strict would want an annotation on every parameter, and `data` is deliberately "Series or DataFrame or dict". One pandas-stub fact worth knowing as a caller: `frame[label]` and `frame[mask]` are typed `Series | DataFrame`, so passing them straight into a plotter fails pyright even though it runs. Two idioms avoid it:

```python
ranked = frame.groupby(["a", "b"])["v"].min().reset_index()       # DataFrame (as_index=False is NOT checkable)
subset = pd.DataFrame(frame[frame["a"] == "x"])                   # wrap a boolean mask
```

Inside the library `frames._one_column` does this narrowing once and **raises on a duplicated label**, which used to produce a wrong chart silently.
