"""The example scripts must keep running: they are the first thing a new user copies."""

import runpy
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[2] / "examples").glob("*.py"))


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs_and_writes_html(script, tmp_path, monkeypatch):
    import plotly.graph_objects as go

    monkeypatch.chdir(tmp_path)  # the scripts write their HTML into the working directory
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)  # never open a browser
    runpy.run_path(str(script), run_name="__main__")
    assert list(tmp_path.glob("*.html")), f"{script.name} wrote nothing"
