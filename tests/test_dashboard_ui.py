"""Ensure the real dashboard renders without losing the trained model."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_loads_and_shows_research_workspace():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60).run()
    assert not app.exception, [str(e.message) for e in app.exception]
    assert len(app.tabs) == 3
    assert any("Malware families." in block.value for block in app.markdown)
    assert any("Experiment at a glance" in block.value for block in app.markdown)
