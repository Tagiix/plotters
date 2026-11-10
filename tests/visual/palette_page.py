"""The palette page: swatches, colour-blind simulations, ΔE tables, ramps, line styles, markers."""

from __future__ import annotations

import html

import plotly.graph_objects as go

import plotters as pl
from plotters import _color as c
from plotters import style


def _swatches(colors: list[str], labels: list[str] | None = None) -> str:
    cells = []
    for i, col in enumerate(colors):
        label = labels[i] if labels else str(i)
        ink = "#000" if c.relative_luminance(col) > 0.4 else "#fff"
        cells.append(
            f'<div class="sw" style="background:{col};color:{ink}"><b>{html.escape(label)}</b>'
            f"<span>{col}</span></div>"
        )
    return f'<div class="swrow">{"".join(cells)}</div>'


def _delta_table(colors: list[str]) -> str:
    cyc = [*colors, colors[0]]
    rows = [
        "<tr><th>vision</th><th>worst adjacent ΔE (incl. wrap)</th><th>worst any pair ΔE (a target for normal vision only)</th></tr>"
    ]
    for name, d in (("normal", None), *((d, d) for d in c.DEFICIENCIES)):
        adj = c.min_adjacent_delta_e(cyc, d)
        pair = c.min_pair_delta_e(colors, d)
        rows.append(f"<tr><td>{name}</td><td>{adj:.1f}</td><td>{pair:.1f}</td></tr>")
    return f"<table>{''.join(rows)}</table>"


def _identity_figure() -> go.Figure:
    fig = go.Figure()
    for i in range(8):
        dash, width = pl.line_style(i + 8)
        fig.add_scatter(
            x=[0, 1],
            y=[8 - i, 8 - i],
            mode="lines",
            showlegend=False,
            line={"color": pl.color(i), "dash": dash, "width": width},
        )
    for i in range(24):
        m = pl.marker(i, size=11)
        fig.add_scatter(
            x=[2 + (i % 8)],
            y=[8 - (i // 8) * 2.4 - 0.5],
            mode="markers",
            showlegend=False,
            marker={**m, "line": {"width": 2, "color": m["color"]}},
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return pl.finish(
        fig,
        "Identity channels",
        "Left: slots 8-15 are dashed. Right: 24 markers, fill changes only after 64",
        height=300,
        legend=False,
    )


def render(dark: bool) -> str:
    with pl.use_style(dark=dark) as st:
        cats = list(st.categorical)
        names = ["blue", "rust", "cyan", "crimson", "amber", "lavender", "green", "olive"]
        sim = "".join(
            f'<div class="simrow"><span>{d}</span>{_swatches([c.simulate(x, d) for x in cats], names)}</div>'
            for d in c.DEFICIENCIES
        )
        fig = _identity_figure()
        fig_html = fig.to_html(
            full_html=False, include_plotlyjs=False, config={"displaylogo": False}
        )
        return (
            f'<div class="panel" style="background:{st.surface};color:{st.ink}">'
            f"<h3>{'Dark' if dark else 'Light'} · surface {st.surface}</h3>"
            f"<h4>Categorical (slot order matters)</h4>{_swatches(cats, names)}"
            f"<h4>As seen with colour-vision deficiencies</h4>{sim}"
            f"<h4>Separation (CIEDE2000)</h4>{_delta_table(cats)}"
            f"<h4>Sequential (one hue)</h4>{_swatches(list(st.sequential))}"
            f"<h4>Diverging (neutral midpoint)</h4>{_swatches(list(st.diverging))}"
            f"<h4>Ink and surface</h4>{_swatches([st.surface, st.grid, st.axis, st.muted, st.ink_2, st.ink], ['surface', 'grid', 'axis', 'muted', 'ink_2', 'ink'])}"
            f"{fig_html}</div>"
        )


def page_section() -> str:
    return (
        '<section id="palette"><h2>Palette</h2>'
        '<p class="checks">What you should see: eight hues that stay apart in every simulated row; '
        "sequential is monotonic light-to-dark (reversed in dark mode); diverging has a grey "
        "midpoint (dark in dark mode); no rainbow.</p>"
        f'<div class="pair">{render(False)}{render(True)}</div></section>'
    )


assert style  # keep the import explicit for readers tracing where the tokens come from
