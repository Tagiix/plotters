# CLAUDE.md: plotters

Context for Claude Code (and any contributor) working in this repository. Read it fully before changing anything.

**What this is.** `plotters` is a generic plotly helper library: pandas in, a themed `go.Figure` out, one consistent colour-blind-safe look, readable docstrings (VS Code hover), and a gallery of every function. It started from a README the author transcribed from photos of an older private library (about 45 functions); that spec was kept as a base and extended to about 110 public functions.

**Non-goals (do not add).** It draws and summarises what the caller gives it. It never fits a model or solves a problem (no lasso/ridge solver, no model fitting; `partial_dependence` only calls a `predict` function you pass). No pie, violin, dual y axis, 3-D, radar, maps or network graphs: they either mislead or need extra dependencies (see "Design rules").

## Commands

```bash
uv venv && uv pip install -e ".[dev]"            # dev environment (hatchling; static editable path, Pylance-friendly)
.venv/bin/ruff check .  &&  .venv/bin/ruff format plotters tests examples
.venv/bin/pyright                                 # typeCheckingMode = "basic"; must report 0 errors
.venv/bin/python -m pytest -q                     # 615 tests, ~15 s
.venv/bin/python -m tests.visual [substring]      # gallery/index.html, or gallery/only.html for matching cards
uv build                                          # wheel + sdist into dist/ (both git-ignored)
scripts/install-into.sh <venv-dir>...             # editable install of this checkout into other venvs
```

**Definition of done for any change:** ruff clean, pyright 0 errors, pytest green, gallery rebuilt, and (for anything visual) the rendered output actually looked at. CI (`.github/workflows/ci.yml`) runs lint, pyright, pytest and the gallery on Python 3.10, 3.11, 3.12 and 3.13 with `fail-fast: false`. After pushing, watch it with `gh run watch <id> --exit-status`; do not assume.

## Snapshot (v0.1.0, 2026-11-10)

119 exports (111 functions), 63 plotly plotters (`plot_*`) + 10 matplotlib print functions in `plotters.static`, 615 tests, 121 gallery cards. Palette worst adjacent colour-blind ΔE 15.6 light / 17.3 dark (floor 8), worst normal-vision pair 26.3 (floor 15). Verified on Python 3.10, 3.12, 3.15.

## Architecture

Flat layout, `plotters/` at the repo root. Every public name is re-exported from `plotters/__init__.py`, which is **generated and grouped by theme** (see "Regenerating `__init__.py`").

Data flow in every plotter: **input normaliser (`frames.py`) → draw traces → `finish()` (`layout.py`)**. Plotters never branch on input shape, and never set title, legend, grid, fonts or colours themselves.

| Module | Role |
|---|---|
| `style.py` | Palette, ramps, `color/line_style/marker`, the plotly template, `set_style/use_style` |
| `layout.py` | `finish()`, `apply_log()`, `log_ticks()`; **all pixel maths lives here** |
| `frames.py` | Input normalisers (`as_series_dict`, `as_value_series`, `pick`), reshaping helpers, `group_summary` |
| `stats.py` | Pure numbers behind the charts (no plotting): acf/pacf/ccf, ljung_box, periodogram, expectile, ROC, one-SE rule... |
| `timeseries.py` | `plot_series` (views, spans, step, range_buttons, drawn), cumulative, drawdown, stacked, rolling, bars_with_line, line_band, calendar_heatmap |
| `analysis.py` | `analyze_series`, `compare_series`, `plot_series_distribution` |
| `diagnostics.py` | gaps, seasonal, lag, ccf, periodogram |
| `correlogram.py` | acf, pacf, acf grid (`partial=True`) |
| `forecasting.py` | forecast, residuals, calibration, quantile calibration, PIT |
| `distributions.py` | null_test, box, histogram, ecdf, qq, cumulative_share, expectile_curve |
| `relationships.py` | scatter (+ `size=` bubbles), shaded_scatter, density2d, regression, scatter_matrix |
| `comparisons.py` | tradeoff (Pareto), rank_heatmap, bump, interval, box_series, small_multiples, paired_difference |
| `categorical.py` | bars (+ `error=`), counts, grouped_bars, lollipop, dumbbell, waterfall, pareto |
| `matrices.py` | matrix, correlation, table_heatmap, contours, table |
| `learning.py` | decision_boundary, coefficient_path, error_curves, roc, partial_dependence, scree, confusion, dendrogram |
| `views.py` | `plot_views`, `annotate_views`, `plot_grid` |
| `static.py` | matplotlib print figures; **opt-in**, not imported by `__init__` |
| `notebook.py`, `io.py` | `show/show_all/autoshow` (hooks `finish`), `save_html/save_png` |
| `_checks.py`, `_color.py`, `_errors.py`, `_types.py` | Private helpers (validation, colour maths + CIEDE2000, `ConventionError`, constants) |

**Five input conventions** (the first argument says which): A series-like (`Series`, wide `DataFrame`, `{label: Series}`), B long frame + column-name strings, C two positional series, D wide/square frame as-is, E `{label: number}` or a Series indexed by label. A wrong shape raises `ConventionError` (a `TypeError`) naming the convention; a missing column raises `KeyError` listing the columns.

## Adding or changing a plotter: checklist

1. **Signature.** Data args, then `title: str = ""`, `subtitle: str = ""`, then `*`, then options, then `height` and `width` (both `int | None`, `width=None` fills the container), `legend` where it makes sense. Use `Literal[...]` for enumerated strings. `series=` is the subset-and-order argument everywhere (including `plot_box`). American spelling is canonical (`color`, `normalize_*`, `center`); British names exist only as function aliases.
2. **Docstring.** Google style: one-line summary, `Args:` documenting **every** parameter (this is what shows on VS Code hover), `Returns:`, `Raises:`. Comments only for the non-obvious *why*. An audit of "every public function has `Args:`" has been run repeatedly; keep it true.
3. **Behaviour.** Normalise input through `frames.py`. Respect the caps (`MAX_SERIES` 8, `MAX_ALL_PAIRS` 3, `MAX_VISIBLE` 20). Refuse rather than mislead. Colours only via `style` (`color(i)`, `get_style()`, `sequential_scale()`); layout only via `finish()`; log axes only via `apply_log`.
4. **Export.** Add to the module's `__all__`, then regenerate `plotters/__init__.py`.
5. **Tests.** A unit test for the behaviour **and** an entry in the `CALLS` table of `tests/unit/test_sizing.py` (a test fails if a plotter is missing from it; it checks `width`/`height` are honoured and the default leaves width unset).
6. **Gallery card.** `tests/visual/cards_*.py`, decorated with `@card(section, title, checks=[...])`; the section must be listed in `ORDER` in `tests/visual/build.py`. The function body is displayed as the example. **Render it and look at it** (headless Chrome screenshot, see below): most layout bugs here were only visible in the picture.
7. **Docs.** Add the function to the README "Which function do I need?" table and its module table, to `CHANGELOG.md`, and log anything notable in the local, git-ignored `plan.md` (the author's working log; it must never be committed or pushed). A mechanical check exists (run it: every public function must appear in the README, and every backticked `plot_*` name in the README must exist).

## Design rules (enforced in code; do not weaken)

- **One y axis, always.** No dual-axis option; use `plot_small_multiples` / `plot_grid` or index to a base.
- **Series caps differ by form, deliberately.** `plot_series`, `plot_cumulative`, `plot_drawdown` and `plot_rolling` unlimited (`drawn=`: 20 visible, rest `legendonly`); other line charts 8 (`MAX_SERIES`; it is deliberately not the palette size); scatter and facets 3 (`MAX_ALL_PAIRS` is a constant, not `len(palette)`); value/rank-coloured and ACF panels uncapped.
- **No violin, no pie.** `plot_box(points=True)` and `plot_bars(normalize=True)` are the honest forms.
- **Colour order is the safety mechanism.** The 15 categorical hues (slots 0-7 the original set, 8-14 added by a search) were searched for colour-blind separation (CIEDE2000 on Machado-simulated protan/deutan/tritan, wrap-around included) and every size from 8 to 15 (`set_style(colors=)`) is validated, so each slot is also CVD-distinct from slot 0. `tests/unit/test_palette.py` enforces it; re-ordering or editing a slot requires re-validation. Worst adjacent CVD ΔE 9.2 / 10.0 (floor 8), worst normal pair 16.2 / 16.0 (floor 15): thin margins, 16 colours failed the floor.
- **Two identity channels that wrap at the same length are one channel.** `marker(i)` shifts the shape after each `lcm(colours, 8)` slots and varies fill after `8 * colours` (192 distinct at every palette size).
- **Cell ink from measured luminance**, never a position threshold (the dark diverging midpoint is dark). Fill-to-ink luminance gap must stay at or above 0.33.
- **Bubble value = area**, not diameter. **Rank colouring only in `plot_rank_heatmap`**; everything else colours by value.
- **Refusals that must keep raising:** negative parts in `plot_stacked`; non-numeric x in `plot_box_series` (linear axis, width from the tightest gap / group count); duplicated column labels; unmatched `group_by_prefix` label; non-Cartesian figures or existing menus/sliders in `plot_views`/`plot_grid`; negative bubble sizes or error bars; Student-t reference with fitted ν ≤ 2.
- **Semantics worth not getting backwards:** a calibration curve *flatter* than the diagonal means *overconfident*; a PIT "U" means overconfident; the acf band **widens** with lag (Bartlett) while the pacf band is flat; `ccf` lag k = corr(x[t], y[t-k]), so a peak at a *negative* lag means y follows x.

## Hard-won technical facts

- **Plotly 7.1 log axes:** `range` is in log10; `tickvals`, trace data and **shapes** are in data units; **annotations are in log10**. (The original spec had this wrong.) Use `apply_log`; `plot_series(markers=, bands=, spans=)` already handle it.
- Use `fig.add_shape` + `fig.add_annotation`, not `add_vline`, for datetime x (the latter breaks on Timestamps).
- Subplot axes are `x`, `x2`, ...; to move a shape/annotation into panel k remap `"paper"` to `"x{k} domain"` (done in `views.plot_grid`).
- Equal-aspect axes need `constrain="domain"`, or plotly stretches the range (calibration, ROC, quantile calibration).
- A **local variable named `width` (or `height`) shadows the parameter**: this silently broke `width=` in six functions (bin width, box width, line width) until `test_sizing.py` caught it. Name such locals `bin_width`, `box_width`, `line_width`.
- Dark mode flips the sequential ramp (deep to bright), so wording like "darker = more" is wrong there. Say "colour = share", "fainter".
- The pandas/plotly **stubs** differ between local and CI and have bitten twice: `fig.data` is typed as its keys (use `list(fig.data)` / `views._traces`), `frame[label]` is `Series | DataFrame`, reductions are `Scalar` (use `stats._f`). Prefer plain numpy where the type is ambiguous.
- Table row height: plotly `Table` clips rows silently if the figure is too short (`150 + 27 * (rows + 1)`).
- `finish()` leaves existing axis titles alone when `xlabel`/`ylabel` is empty (needed by `plot_views`); legend moves to a scrolling right column past 12 entries; a dropdown or range buttons reserve extra top margin (`menu=True`).

## Tooling gotchas (these cost real time)

- **`ruff format` reflows code**, so a script that string-replaces source it wrote earlier may silently not match. Re-read the file first, or use an `ast`/whitespace-tolerant regex. `# fmt: skip` is used on a few dense lines.
- `ast` column offsets are **UTF-8 byte offsets**; convert to character indexes when editing text containing characters like `×`, or inserts land in the wrong place.
- **Regenerating `__init__.py`:** it imports every module, so a broken `__init__` cannot generate itself. Write a one-line bootstrap `__init__` first, then generate (group order: theme and layout, data shaping and stats, time series, distributions, relationships, comparisons, categories and matrices, learning, views).
- macOS: `sed -i ''` needs the empty argument (prefer Python for edits). The user's shell aliases `ls`, which can hang in a script: use `command ls`. Long `sleep` is blocked: use `gh run watch` or a Monitor.
- `git filter-branch` checks out the rewritten tree, which **deletes removed tracked files from disk**, and `gc --prune=now` makes them unrecoverable. Copy anything you might want back first.
- Tests must never call `fig.show()` (opens a browser): `tests/unit/test_examples.py` monkeypatches it. Matplotlib tests use Agg via `tests/conftest.py`.

**Visual check recipe** (what found most layout bugs): `python -m tests.visual <name>` writes `gallery/only.html`; then headless Chrome for Testing (installed by `plotly_get_chrome` into `~/Library/Application Support/choreographer/deps/...`):
`"<chrome>" --headless=new --hide-scrollbars --window-size=1700,900 --virtual-time-budget=6000 --screenshot=out.png file:///.../gallery/only.html`, then read the PNG. Fragment URLs (`#slug`) render blank; build a single-card page instead.

## Git, GitHub, releases

- Remote `origin` = `git@github.com:Tagiix/plotters.git`, branch `main`, currently **private**. Commits are authored by the user's configured git identity; end commit messages with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- **Do not push, create or publish anything (repo, tag, release, public visibility, Pages) unless asked.** The repo was created and pushed once at the user's explicit request.
- Never commit: `plan.md` (local working log, kept on disk but git-ignored and purged from history), `gallery/`, `dist/`, `.venv/`, `.DS_Store`, any `*.zip`, personal photos or paths. The history was rewritten once to purge an 84 MB zip.
- Release: bump `version` in `pyproject.toml` (single source; `__version__` reads package metadata), update `CHANGELOG.md`, tag `vX.Y.Z`, push the tag, then verify an install from `git+ssh://git@github.com/Tagiix/plotters.git@vX.Y.Z` in a **fresh** venv.
- The package must ship `plotters/py.typed` (without it Pylance/pyright in other projects ignore the types). Check the wheel contains it after any packaging change.

## Deferred ideas (not built)

`set_style(palette=...)` with mandatory re-validation; light/dark toggle inside exported HTML; a `pl.gallery()` CLI entry point; pixel-snapshot tests; `plot_seasonal`-style decomposition (needs an algorithm); GitHub Pages for the gallery; `width=` is now everywhere, but `static` figures size via `figsize`/`figure_size()` instead. Before making the repo public: decide whether the commit email should be a GitHub no-reply address (it is in every commit and in `pyproject.toml`).
