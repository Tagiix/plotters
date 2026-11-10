"""Shared type aliases and limits."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

SeriesLike = pd.Series | pd.DataFrame | Mapping[str, pd.Series]
"""Convention A input: a Series, a wide DataFrame (one series per column) or ``{label: Series}``."""

MAX_VISIBLE = 20
"""``plot_series``: series drawn on open; the rest start ``legendonly``."""

MAX_SERIES = 8
"""Colour guard for line charts other than ``plot_series``: one series per categorical colour."""

MAX_ALL_PAIRS = 3
"""Scatter and facets are read pair by pair; no ordering of 8 hues clears the floors past 3."""
