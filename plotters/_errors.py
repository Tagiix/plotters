"""Errors raised when the wrong input convention is passed."""

from __future__ import annotations

from typing import Any


class ConventionError(TypeError):
    """The data passed has the wrong shape for this plotter's input convention.

    It means "wrong convention", not "bad data": see the four conventions in the README.
    """


_HINTS = {
    "A": "a Series, a wide DataFrame (one series per column) or a {label: Series} dict",
    "B": "a long (tidy) DataFrame followed by column-name strings (x=, y=, ...)",
    "C": "two positional series: bars (a Series) and line (Series or dict)",
    "D": "a wide/square DataFrame whose rows and columns are the two axes",
}


def convention_error(function: str, convention: str, got: Any, advice: str = "") -> ConventionError:
    """Build a :class:`ConventionError` that says what was expected and what arrived."""
    message = (
        f"{function} expects {_HINTS[convention]} (convention {convention}); "
        f"got {type(got).__name__}."
    )
    return ConventionError(f"{message} {advice}".strip())
