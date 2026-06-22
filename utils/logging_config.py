"""Centralized logging configuration for the risk calculator."""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

# Ensure UTF-8 capable streams on Windows consoles to avoid encoding errors.
# Skip this when running under pytest to avoid interfering with test capture.
if (
    sys.platform == "win32"
    and "pytest" not in sys.modules
    and not os.environ.get("PYTEST_CURRENT_TEST")
):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # Already wrapped or in a non-standard environment
        pass


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    include_timestamp: bool = False,
) -> logging.Logger:
    """
    Configure logging for the risk calculator.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to write logs to file
        include_timestamp: Include timestamps in log output. Disabled by default
            to keep pipeline logs reproducible across runs.

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("gastric_risk")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    console_pattern = "%(levelname)-8s | %(message)s"
    if include_timestamp:
        console_pattern = "%(asctime)s | " + console_pattern

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        console_pattern,
        datefmt="%Y-%m-%d %H:%M:%S" if include_timestamp else None,
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_pattern = "%(levelname)-8s | %(name)s | %(message)s"
        if include_timestamp:
            file_pattern = "%(asctime)s | " + file_pattern

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            file_pattern,
            datefmt="%Y-%m-%d %H:%M:%S" if include_timestamp else None,
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Get the configured logger instance."""
    return logging.getLogger("gastric_risk")
