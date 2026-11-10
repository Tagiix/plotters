"""Shared type aliases and limits."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

SeriesLike = pd.Series | pd.DataFrame | Mapping[str, pd.Series]
"""Convention A input: a Series, a wide DataFrame (one series per column) or ``{label: Series}``."""

MAX_VISIBLE = 20
"""``plot_series``: series drawn on open; the rest start ``legendonly``."""

MAX_SERIES = 8
"""``plot_series(views=)``: past this many unnamed (baseline) series the views become an explicit allowlist."""
