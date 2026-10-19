"""Structured logging configuration for Notely.

Follows privacy principles:
- Never logs complete document texts or full prompts.
- Never logs API keys or authorization headers.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Obtain a named logger configured with a privacy-conscious, standardized format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
