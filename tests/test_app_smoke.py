"""Smoke tests for the optional Streamlit demo (app.py).

The interactive demo is not part of the headless analytical pipeline, but it
should never be allowed to silently rot. These tests guarantee that the module
stays syntactically valid and, when Streamlit is installed, that the script
executes end-to-end in a simulated runtime without raising.
"""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_app_is_syntactically_valid():
    """app.py must always compile (catches syntax/indentation regressions)."""
    assert APP_PATH.exists(), "app.py is missing"
    # Raises py_compile.PyCompileError on failure.
    py_compile.compile(str(APP_PATH), doraise=True)


def test_app_runs_in_simulated_runtime():
    """app.py executes top-to-bottom under Streamlit's AppTest without error."""
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")

    app = streamlit_testing.AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()

    # A clean run produces no uncaught exceptions surfaced to the UI.
    assert not app.exception, f"app.py raised in simulated runtime: {app.exception}"
