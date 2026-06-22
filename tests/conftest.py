"""Pytest configuration for test discovery and path setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Mark that we're running under pytest to prevent stdout/stderr wrapping issues
os.environ["PYTEST_CURRENT_TEST"] = "true"

# Force a non-GUI Matplotlib backend for headless/CI environments
os.environ.setdefault("MPLBACKEND", "Agg")

# Add the repository root to sys.path for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Ensure joblib has a writable temp directory to avoid permission warnings
JOBLIB_TEMP = ROOT_DIR / ".joblib_temp"
JOBLIB_TEMP.mkdir(exist_ok=True)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(JOBLIB_TEMP))
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
