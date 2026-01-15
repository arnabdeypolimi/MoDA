"""
Reusable logging utilities with custom formatting.

This module provides a consistent logging setup across all scripts with the format:
[[LEVEL]] - date - time - message

Example:
    [[INFO]] - 2025-12-11 14:30:45 - Loading configuration...
    [[WARNING]] - 2025-12-11 14:30:46 - Deprecated feature used
    [[ERROR]] - 2025-12-11 14:30:47 - Failed to load file

Usage:
    # Option 1: Quick setup with module-level logger
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("This is an info message")
    
    # Option 2: Use the convenience root logger
    from src.utils.logger import configure_root_logger, log_info, log_warning, log_error
    configure_root_logger(level="DEBUG")
    log_info("Starting application...")
"""

import logging
import sys
from typing import ClassVar

__all__ = [
    "CustomFormatter",
    "setup_logger", 
    "configure_root_logger",
    "get_logger",
    # Convenience functions for module-level logging
    "log_debug",
    "log_info", 
    "log_warning",
    "log_error",
    "log_critical",
]


class CustomFormatter(logging.Formatter):
    """
    Custom formatter with [[LEVEL]] format and optional colors.

    Format: [[LEVEL]] - YYYY-MM-DD HH:MM:SS - message
    """

    # ANSI color codes
    COLORS : ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET : ClassVar[str] = "\033[0m"

    def __init__(self, use_colors: bool = True, datefmt: str = "%Y-%m-%d %H:%M:%S"):
        """
        Initialize the custom formatter.

        Args:
            use_colors: If True, use ANSI colors for log levels (disable for file logging)
            datefmt: Date format string for timestamps
        """
        self.use_colors = use_colors
        self.datefmt = datefmt
        super().__init__(datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with custom formatting."""
        if self.use_colors:
            color = self.COLORS.get(record.levelname, self.RESET)
            log_fmt = f"{color}[[{record.levelname}]]{self.RESET} - %(asctime)s - %(message)s"
        else:
            log_fmt = f"[[{record.levelname}]] - %(asctime)s - %(message)s"

        formatter = logging.Formatter(log_fmt, datefmt=self.datefmt)
        return formatter.format(record)


def setup_logger(
    name: str| None = None,
    level: str = "INFO",
    use_colors: bool = True,
    log_file: str| None = None,
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Set up and configure a logger with custom formatting.

    Args:
        name: Logger name (use __name__ for module-specific loggers, None for root logger)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_colors: If True, use ANSI colors in console output
        log_file: Optional path to log file (colors disabled for file output)
        datefmt: Date format string for timestamps

    Returns:
        Configured logger instance

    Example:
        >>> from experimental.projects.improve_synthesis.logging_utils import setup_logger
        >>> logger = setup_logger(__name__, level="DEBUG")
        >>> logger.info("This is an info message")
        [[INFO]] - 2025-12-11 14:30:45 - This is an info message

        >>> # With file output
        >>> logger = setup_logger(__name__, log_file="train.log")
        >>> logger.warning("This goes to both console and file")
    """
    # Get or create logger
    logger = logging.getLogger(name)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(CustomFormatter(use_colors=use_colors, datefmt=datefmt))
    logger.addHandler(console_handler)

    # File handler without colors (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(CustomFormatter(use_colors=False, datefmt=datefmt))
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    return logger


def configure_root_logger(
    level: str = "INFO",
    use_colors: bool = True,
    log_file: str| None = None,
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """
    Configure the root logger with custom formatting.

    This affects all loggers in the application unless they have their own handlers.
    Use this for simple scripts or when you want consistent formatting everywhere.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_colors: If True, use ANSI colors in console output
        log_file: Optional path to log file (colors disabled for file output)
        datefmt: Date format string for timestamps

    Example:
        >>> from experimental.projects.improve_synthesis.logging_utils import configure_root_logger
        >>> import logging
        >>>
        >>> configure_root_logger(level="DEBUG", log_file="app.log")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Root logger configured!")
        [[INFO]] - 2025-12-11 14:30:45 - Root logger configured!
    """
    # Get root logger
    root_logger = logging.getLogger()

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(CustomFormatter(use_colors=use_colors, datefmt=datefmt))
    root_logger.addHandler(console_handler)

    # File handler without colors (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(CustomFormatter(use_colors=False, datefmt=datefmt))
        root_logger.addHandler(file_handler)


def _get_log_level_from_env() -> str:
    """Get log level from environment variable, defaulting to INFO."""
    import os
    return os.environ.get("MODA_LOG_LEVEL", "INFO").upper()


# Convenience function for quick setup
def get_logger(
    name: str,
    level: str | None = None,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Quick and simple logger setup for most use cases.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (defaults to MODA_LOG_LEVEL env var, or INFO)
        use_colors: If True, use ANSI colors

    Returns:
        Configured logger instance

    Environment Variables:
        MODA_LOG_LEVEL: Set to DEBUG, INFO, WARNING, ERROR, or CRITICAL

    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Quick setup!")
        [[INFO]] - 2025-12-11 14:30:45 - Quick setup!
        
        # Run with debug logging:
        # MODA_LOG_LEVEL=DEBUG python your_script.py
    """
    if level is None:
        level = _get_log_level_from_env()
    return setup_logger(name=name, level=level, use_colors=use_colors)


# =============================================================================
# Module-level convenience functions
# =============================================================================

# Default module logger (configured on first use)
_default_logger: logging.Logger | None = None


def _get_default_logger() -> logging.Logger:
    """Get or create the default module-level logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger(name="moda", level="INFO", use_colors=True)
    return _default_logger


def log_debug(message: str, *args, **kwargs) -> None:
    """Log a debug message using the default logger."""
    _get_default_logger().debug(message, *args, **kwargs)


def log_info(message: str, *args, **kwargs) -> None:
    """Log an info message using the default logger."""
    _get_default_logger().info(message, *args, **kwargs)


def log_warning(message: str, *args, **kwargs) -> None:
    """Log a warning message using the default logger."""
    _get_default_logger().warning(message, *args, **kwargs)


def log_error(message: str, *args, **kwargs) -> None:
    """Log an error message using the default logger."""
    _get_default_logger().error(message, *args, **kwargs)


def log_critical(message: str, *args, **kwargs) -> None:
    """Log a critical message using the default logger."""
    _get_default_logger().critical(message, *args, **kwargs)
