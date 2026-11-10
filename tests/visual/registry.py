"""Card registry: each gallery card is a function that returns a figure.

The function body is shown on the page as the exact call, so the example can never drift
from what is drawn. The docstring-free ``checks`` list is the "what you should see" list.
"""

from __future__ import annotations

import inspect
import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass

import plotly.graph_objects as go


@dataclass
class Card:
    section: str
    title: str
    build: Callable[[], go.Figure]
    checks: list[str]

    @property
    def source(self) -> str:
        lines = textwrap.dedent(inspect.getsource(self.build)).splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("def "))
        return textwrap.dedent("\n".join(lines[start + 1 :])).strip()

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9_]+", "-", f"{self.section}-{self.title}".lower()).strip("-")


CARDS: list[Card] = []


def card(section: str, title: str, checks: list[str]):
    """Register a function returning a ``Figure`` as a gallery card."""

    def wrap(fn: Callable[[], go.Figure]) -> Callable[[], go.Figure]:
        CARDS.append(Card(section, title, fn, checks))
        return fn

    return wrap
