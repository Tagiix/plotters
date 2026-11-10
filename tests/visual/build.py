"""Build ``gallery/index.html``: every card, light and dark side by side."""

from __future__ import annotations

import base64
import html
import io
import traceback
from pathlib import Path

import matplotlib as mpl
from plotly.offline import get_plotlyjs

import plotters as pl

from . import (  # noqa: F401  (registers cards)
    cards_analysis,
    cards_categorical,
    cards_comparisons,
    cards_core,
    cards_distributions,
    cards_matrices,
    cards_static,
)
from .palette_page import page_section
from .registry import CARDS, Card

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "gallery"

CSS = """
:root{--bg:#fbfbfa;--ink:#14171c;--muted:#6b7280;--line:#e2e4e8;--code:#f1f2f4;--accent:#2b5cda}
@media (prefers-color-scheme:dark){:root{--bg:#0d0f13;--ink:#e6e8ec;--muted:#8b93a1;--line:#242932;--code:#171a21;--accent:#6b97f5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,-apple-system,Segoe UI,sans-serif}
nav{position:fixed;inset:0 auto 0 0;width:230px;overflow:auto;padding:20px 16px;border-right:1px solid var(--line)}
nav h1{font-size:16px;margin:0 0 12px}nav a{display:block;color:var(--muted);text-decoration:none;padding:2px 0;font-size:13px}
nav a:hover{color:var(--accent)}nav .sec{margin-top:14px;font-weight:600;color:var(--ink);font-size:13px}
main{margin-left:230px;padding:24px 28px 80px;max-width:1900px}
section{margin-bottom:44px}h2{font-size:22px;border-bottom:1px solid var(--line);padding-bottom:6px}
.card{margin:26px 0 34px}.card h3{margin:0 0 4px;font-size:16px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.panel{border:1px solid var(--line);border-radius:8px;padding:10px;min-width:0;overflow:hidden}
.checks{margin:6px 0 10px;color:var(--muted);font-size:13px}.checks li{margin:1px 0}
pre{background:var(--code);padding:10px 12px;border-radius:6px;overflow:auto;font:12.5px/1.45 ui-monospace,Menlo,monospace;margin:8px 0 10px}
.err{border:1px solid #c33;color:#c33;padding:8px;border-radius:6px;white-space:pre-wrap;font:12px monospace}
.swrow{display:flex;flex-wrap:wrap}.sw{width:76px;height:54px;padding:5px;font-size:11px;display:flex;flex-direction:column;justify-content:space-between}
.sw span{font-family:ui-monospace,monospace;font-size:10px;opacity:.85}.simrow{display:flex;align-items:center;gap:8px}.simrow>span{width:52px;font-size:12px}
table{border-collapse:collapse;font-size:13px}td,th{padding:3px 10px;border-bottom:1px solid rgba(128,128,128,.3);text-align:left}
h4{margin:14px 0 6px;font-size:13px}@media(max-width:1100px){.pair{grid-template-columns:1fr}nav{display:none}main{margin-left:0}}
"""


def _render(card: Card, dark: bool) -> str:
    try:
        with pl.use_style(dark=dark), mpl.rc_context():
            fig = card.build()
            if hasattr(fig, "savefig"):  # a matplotlib print figure: embed it as a PNG
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=170)
                import matplotlib.pyplot as plt

                plt.close(fig)
                data = base64.b64encode(buf.getvalue()).decode()
                return f'<img alt="{html.escape(card.title)}" style="max-width:100%" src="data:image/png;base64,{data}">'
            return fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={"displaylogo": False, "responsive": True},
            )
    except Exception:
        return f'<div class="err">{html.escape(traceback.format_exc())}</div>'


def build(out: Path = OUT, only: str | None = None) -> Path:
    """Write the gallery and return the path to ``index.html``.

    Args:
        out: Output folder.
        only: A substring of a card slug; builds a page with just the matching cards.
    """
    out.mkdir(parents=True, exist_ok=True)
    (out / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")

    sections: dict[str, list[Card]] = {}
    for card in CARDS:
        if only and only not in card.slug:
            continue
        sections.setdefault(card.section, []).append(card)

    nav = ['<h1>plotters gallery</h1><a href="#palette">Palette</a>']
    body = [] if only else [page_section()]
    for name, cards in sections.items():
        nav.append(f'<div class="sec">{html.escape(name)}</div>')
        body.append(f"<section><h2>{html.escape(name)}</h2>")
        for card in cards:
            nav.append(f'<a href="#{card.slug}">{html.escape(card.title)}</a>')
            checks = "".join(f"<li>{html.escape(t)}</li>" for t in card.checks)
            body.append(
                f'<div class="card" id="{card.slug}"><h3>{html.escape(card.title)}</h3>'
                f'<ul class="checks">{checks}</ul><pre>{html.escape(card.source)}</pre>'
                f'<div class="pair"><div class="panel">{_render(card, False)}</div>'
                f'<div class="panel">{_render(card, True)}</div></div></div>'
            )
        body.append("</section>")

    page = (
        '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>plotters gallery</title>'
        f'<script src="plotly.min.js"></script><style>{CSS}</style></head><body>'
        f"<nav>{''.join(nav)}</nav><main>{''.join(body)}</main></body></html>"
    )
    index = out / ("index.html" if not only else "only.html")
    index.write_text(page, encoding="utf-8")
    return index
