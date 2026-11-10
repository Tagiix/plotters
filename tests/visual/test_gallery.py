"""Smoke test: every gallery card builds a figure in both themes."""

import plotly.graph_objects as go
import pytest

import plotters as pl

from . import cards_analysis, cards_comparisons, cards_core, cards_distributions, cards_matrices  # noqa: F401  (registers the cards)
from .registry import CARDS


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
@pytest.mark.parametrize("card", CARDS, ids=lambda c: c.slug)
def test_card_builds(card, dark):
    with pl.use_style(dark=dark):
        fig = card.build()
    assert isinstance(fig, go.Figure)
    assert card.checks, "every card needs a 'what you should see' list"
