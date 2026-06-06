"""
Structured logging for MODEL_GENERATOR_V2.

Provides a consistent logging interface across all modules with
colored console output, structured formatting, and configurable levels.

Dependencies:
    - logging (stdlib)

Classes:
    ColoredFormatter: Custom formatter with ANSI color support.

Functions:
    get_logger: Factory for module-specific loggers.
    setup_logging: Configure global logging level and format.
"""

import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom log formatter with ANSI color codes for console output.

    Attributes:
        COLORS: Mapping of log levels to ANSI color escape codes.
        RESET: ANSI code to reset terminal colors.
    """

    COLORS = {
        logging.DEBUG: "\033[36m",      # Cyan
        logging.INFO: "\033[32m",       # Green
        logging.WARNING: "\033[33m",    # Yellow
        logging.ERROR: "\033[31m",      # Red
        logging.CRITICAL: "\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with color based on severity level.

        Args:
            record: The log record to format.

        Returns:
            Formatted log string with ANSI color codes.
        """
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


_loggers: dict[str, logging.Logger] = {}
_default_level = logging.INFO


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get or create a named logger with console output.

    Creates a logger with a consistent format. Loggers are cached
    to prevent duplicate handler attachment on repeated calls.

    Args:
        name: Logger name, typically the module path
              (e.g., 'model_generator_v2.core.pipeline').
        level: Optional override for the logging level.
               Defaults to the global level set by setup_logging().

    Returns:
        Configured logging.Logger instance.

    Example:
        >>> logger = get_logger('model_generator_v2.core')
        >>> logger.info("Pipeline initialized")
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level or _default_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level or _default_level)

        formatter = ColoredFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    _loggers[name] = logger
    return logger


def setup_logging(level: int = logging.INFO) -> None:
    """Configure global logging level for all MODEL_GENERATOR_V2 loggers.

    Args:
        level: The logging level (e.g., logging.DEBUG, logging.INFO).

    Example:
        >>> setup_logging(logging.DEBUG)
    """
    global _default_level
    _default_level = level
    for logger in _loggers.values():
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
