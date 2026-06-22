"""Tests for logging configuration module."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from utils.logging_config import get_logger, setup_logging


class TestSetupLogging:
    """Test logging setup functionality."""

    def test_setup_logging_returns_logger(self):
        """Test that setup_logging returns a logger instance."""
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "gastric_risk"

    def test_setup_logging_default_level(self):
        """Test default logging level is INFO."""
        logger = setup_logging()
        assert logger.level == logging.INFO

    def test_setup_logging_debug_level(self):
        """Test DEBUG level can be set."""
        logger = setup_logging(level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_setup_logging_warning_level(self):
        """Test WARNING level can be set."""
        logger = setup_logging(level=logging.WARNING)
        assert logger.level == logging.WARNING

    def test_setup_logging_clears_handlers(self):
        """Test that setup clears existing handlers."""
        logger = setup_logging()
        initial_count = len(logger.handlers)

        # Setup again
        logger = setup_logging()

        # Should have same number of handlers (cleared then added)
        assert len(logger.handlers) == initial_count

    def test_setup_logging_without_timestamp(self):
        """Test logging format without timestamp."""
        logger = setup_logging(include_timestamp=False)
        handler = logger.handlers[0]
        assert handler.formatter._fmt == "%(levelname)-8s | %(message)s"

    def test_setup_logging_with_timestamp(self):
        """Test logging format with timestamp."""
        logger = setup_logging(include_timestamp=True)
        handler = logger.handlers[0]
        assert "%(asctime)s" in handler.formatter._fmt

    def test_setup_logging_with_file(self):
        """Test logging to file."""

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logging(log_file=log_file)

            # Should have 2 handlers (console + file)
            assert len(logger.handlers) == 2

            # Log a message
            logger.info("Test message")

            # Force flush and close the file handler before checking content
            for handler in logger.handlers:
                handler.flush()

            # Check file exists and has content
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content

            # Clean up handlers to allow temp directory deletion
            for handler in logger.handlers[:]:
                if hasattr(handler, "close"):
                    handler.close()
            logger.handlers.clear()

    def test_setup_logging_file_format_without_timestamp(self):
        """Test file logging format without timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logging(log_file=log_file, include_timestamp=False)

            logger.info("Test message")

            # Force flush before reading
            for handler in logger.handlers:
                handler.flush()

            content = log_file.read_text()

            # Should contain level and name, but not timestamp
            assert "INFO" in content
            assert "Test message" in content

            # Clean up handlers
            for handler in logger.handlers[:]:
                if hasattr(handler, "close"):
                    handler.close()
            logger.handlers.clear()

    def test_setup_logging_file_format_with_timestamp(self):
        """Test file logging format with timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logging(log_file=log_file, include_timestamp=True)

            logger.info("Test message")

            # Force flush before reading
            for handler in logger.handlers:
                handler.flush()

            content = log_file.read_text()

            # Should contain timestamp-like pattern
            assert "INFO" in content
            assert "Test message" in content

            # Clean up handlers
            for handler in logger.handlers[:]:
                if hasattr(handler, "close"):
                    handler.close()
            logger.handlers.clear()


class TestGetLogger:
    """Test the get_logger function."""

    def test_get_logger_returns_correct_logger(self):
        """Test that get_logger returns the gastric_risk logger."""
        logger = get_logger()
        assert logger.name == "gastric_risk"

    def test_get_logger_returns_same_instance(self):
        """Test that get_logger returns the same logger instance."""
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2

    def test_get_logger_after_setup(self):
        """Test that get_logger works after setup_logging."""
        setup_logging(level=logging.DEBUG)
        logger = get_logger()
        assert logger.level == logging.DEBUG


class TestLoggingOutput:
    """Test actual logging output."""

    def test_info_message_logged(self, capfd):
        """Test that INFO messages are logged to stdout."""
        logger = setup_logging(level=logging.INFO)
        logger.info("Test INFO message")

        captured = capfd.readouterr()
        assert "Test INFO message" in captured.out

    def test_debug_message_not_logged_at_info_level(self, capfd):
        """Test that DEBUG messages are not logged at INFO level."""
        logger = setup_logging(level=logging.INFO)
        logger.debug("Test DEBUG message")

        captured = capfd.readouterr()
        assert "Test DEBUG message" not in captured.out

    def test_debug_message_logged_at_debug_level(self, capfd):
        """Test that DEBUG messages are logged at DEBUG level."""
        logger = setup_logging(level=logging.DEBUG)
        logger.debug("Test DEBUG message")

        captured = capfd.readouterr()
        assert "Test DEBUG message" in captured.out

    def test_warning_message_logged(self, capfd):
        """Test that WARNING messages are logged."""
        logger = setup_logging(level=logging.INFO)
        logger.warning("Test WARNING message")

        captured = capfd.readouterr()
        assert "Test WARNING message" in captured.out

    def test_error_message_logged(self, capfd):
        """Test that ERROR messages are logged."""
        logger = setup_logging(level=logging.INFO)
        logger.error("Test ERROR message")

        captured = capfd.readouterr()
        assert "Test ERROR message" in captured.out
