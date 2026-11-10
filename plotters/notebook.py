"""Notebook helpers.

Plotters return a ``Figure`` and draw nothing, so Jupyter renders only the *last expression
in a cell*; two calls in one cell show one chart. Either call :func:`show` / :func:`show_all`
explicitly or switch on :func:`autoshow`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import plotly.graph_objects as go

__all__ = ["show", "show_all", "autoshow"]

_autoshow = False


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
    except ImportError:
        return False
    return get_ipython() is not None


def show(*figures: go.Figure) -> None:
    """Display figures in order, so several can appear from one notebook cell.

    Outside a notebook this opens each figure with ``fig.show()``.

    Args:
        *figures: Plotly figures.

    Example:
        >>> pl.show(pl.plot_series(a, "A"), pl.plot_series(b, "B"))
    """
    if _in_notebook():
        from IPython.display import display  # type: ignore[import-not-found]

        for fig in figures:
            display(fig)
    else:
        for fig in figures:
            fig.show()


def show_all(figures: Mapping[str, go.Figure] | Iterable[go.Figure]) -> None:
    """Display every figure of a dict or list, e.g. ``analyze_series(...)["figures"]``.

    Args:
        figures: ``{name: Figure}`` or an iterable of figures.
    """
    show(*(figures.values() if isinstance(figures, Mapping) else figures))


def autoshow(enabled: bool = True) -> None:
    """Display each figure as soon as its plotter finishes, even mid-cell.

    The figure is still returned. Because it displays on creation, end a cell with ``;`` or
    assign the result (``fig = pl.plot_series(...)``) to avoid Jupyter showing it twice, and
    note that edits made to the figure afterwards are not reflected in what was shown.
    Applies to every plotter ending in :func:`plotters.finish`, including your own.
    Only acts inside IPython / Jupyter; elsewhere nothing is shown.

    Args:
        enabled: ``False`` switches it off again.
    """
    global _autoshow
    _autoshow = enabled


def _maybe_display(fig: go.Figure) -> None:
    if _autoshow and _in_notebook():
        from IPython.display import display  # type: ignore[import-not-found]

        display(fig)
