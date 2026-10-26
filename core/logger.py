"""
logger.py — Centralised logging for AskMyDocs.

Creates a rotating file handler (app.log) and a stream handler (console).
Also exposes an in-memory StringIO buffer that the Streamlit Logs tab can
read without touching the filesystem.

Usage
-----
    from core.logger import get_logger
    log = get_logger(__name__)
    log.info("Something happened")
"""

from __future__ import annotations

import io
import logging
import sys
from logging.handlers import RotatingFileHandler

import config

# ---------------------------------------------------------------------------
# Shared in-memory buffer — the Streamlit Logs tab reads from this
# ---------------------------------------------------------------------------
_log_buffer: io.StringIO = io.StringIO()

# Guard so we only configure the root logger once even if this module is
# imported multiple times (e.g. during pytest collection).
_configured: bool = False


def _configure_root_logger() -> None:
    """Set up file, console, and in-memory handlers on the root logger."""
    global _configured  # noqa: PLW0603
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # -- rotating file handler ------------------------------------------------
    try:
        file_handler = RotatingFileHandler(
            filename=config.LOG_FILE,
            maxBytes=config.LOG_MAX_BYTES,
            backupCount=config.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError as exc:
        # Non-fatal: log to console only if we can't write the file.
        print(f"[WARN] Could not open log file {config.LOG_FILE}: {exc}", file=sys.stderr)

    # -- console handler -------------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    # -- in-memory buffer handler (for Streamlit Logs tab) --------------------
    mem_handler = logging.StreamHandler(_log_buffer)
    mem_handler.setLevel(logging.DEBUG)
    mem_handler.setFormatter(fmt)
    root.addHandler(mem_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring the root logger is configured first.

    Parameters
    ----------
    name:
        Usually ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        A named child logger inheriting root-level handlers.
    """
    _configure_root_logger()
    return logging.getLogger(name)


def get_log_contents() -> str:
    """Return everything written to the in-memory log buffer so far.

    Used by the Streamlit Logs tab to display recent log output.
    """
    return _log_buffer.getvalue()


def clear_log_buffer() -> None:
    """Clear the in-memory log buffer (e.g. on session reset)."""
    _log_buffer.truncate(0)
    _log_buffer.seek(0)


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _log = get_logger("logger_test")
    _log.debug("Debug message")
    _log.info("Info message")
    _log.warning("Warning message")
    _log.error("Error message")
    print("\n--- In-memory buffer contents ---")
    print(get_log_contents())
