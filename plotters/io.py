"""Saving figures."""

from __future__ import annotations

from os import PathLike

import plotly.graph_objects as go

__all__ = ["save_html", "save_png"]


def save_html(fig: go.Figure, path: str | PathLike[str], *, cdn: bool = False) -> None:
    """Write an interactive, self-contained HTML file.

    Args:
        fig: Figure to save.
        path: Output file.
        cdn: Load plotly.js from a CDN instead of embedding it: about 3 MB smaller but needs
            an internet connection to open.
    """
    fig.write_html(str(path), include_plotlyjs="cdn" if cdn else True)


def save_png(fig: go.Figure, path: str | PathLike[str], *, scale: float = 2.0) -> None:
    """Write a PNG through kaleido.

    Needs ``kaleido`` and a Chrome install (``plotly_get_chrome``); on a headless machine it
    often fails. ``plotters.static`` draws the print figures with matplotlib instead.

    Args:
        fig: Figure to save.
        path: Output file.
        scale: Resolution multiplier.

    Raises:
        RuntimeError: kaleido or Chrome is unavailable.
    """
    try:
        fig.write_image(str(path), scale=scale)
    except Exception as exc:  # kaleido raises several unrelated types when Chrome is missing
        raise RuntimeError(
            "PNG export needs kaleido and Chrome (run `plotly_get_chrome`). "
            "For print figures use plotters.static, which only needs matplotlib."
        ) from exc
