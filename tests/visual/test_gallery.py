"""Smoke test: every gallery card builds a figure in both themes."""

import matplotlib as mpl
import plotly.graph_objects as go
import pytest

import plotters as pl

from . import (  # noqa: F401  (registers the cards)
    cards_analysis,
    cards_categorical,
    cards_comparisons,
    cards_core,
    cards_distributions,
    cards_forecasting,
    cards_learning,
    cards_matrices,
    cards_static,
)
from .registry import CARDS


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
@pytest.mark.parametrize("card", CARDS, ids=lambda c: c.slug)
def test_card_builds(card, dark):
    with pl.use_style(dark=dark), mpl.rc_context():
        fig = card.build()
    assert isinstance(fig, go.Figure) or hasattr(fig, "savefig")
    assert card.checks, "every card needs a 'what you should see' list"
