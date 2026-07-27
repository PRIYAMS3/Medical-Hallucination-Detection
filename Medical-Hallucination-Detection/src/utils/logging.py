"""Logging utilities for reproducible research runs."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(
    log_dir: str | Path,
    file_name: str = "training.log",
    level: str = "INFO",
    log_format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Configure project logging to both console and file.

    Args:
        log_dir: Directory where log files should be created.
        file_name: Name of the log file.
        level: Logging level, such as ``INFO`` or ``DEBUG``.
        log_format: Format string for log messages.
        date_format: Datetime format used in log records.

    Returns:
        Configured root logger.
    """
    resolved_log_dir = Path(log_dir)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    log_file = resolved_log_dir / file_name

    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
