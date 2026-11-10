"""Smallest useful tour: theme once, then one call per chart. Run: python examples/quickstart.py"""

import numpy as np
import pandas as pd

import plotters as pl

rng = np.random.default_rng(0)
dates = pd.date_range("2023-01-02", periods=250, freq="B")
returns = pd.DataFrame(rng.standard_t(5, (250, 3)) * 0.008, index=dates, columns=["a", "b", "c"])

pl.set_style(dark=False)  # once, at the top

fig = pl.plot_series(returns, "Daily returns", "Three models", percent=True, range_buttons=True)
pl.save_html(fig, "quickstart.html")  # interactive and self-contained

curve = pl.plot_cumulative(returns, "Growth of 1", compound=True)
drawdown = pl.plot_drawdown(returns, "Drawdown")
pl.save_html(pl.plot_grid({"Growth": curve, "Drawdown": drawdown}), "quickstart_grid.html")
# In a notebook, pl.show(curve, drawdown) displays both from one cell.
