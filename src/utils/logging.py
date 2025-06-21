"""
Centralized logging module for Astrabot.

This module provides:
- Singleton logger with consistent configuration
- Automatic sensitive data masking
- Structured logging support
- Performance tracking decorators
- Thread-safe logging
- File rotation support
"""

import functools
import json
import logging
import logging.handlers
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

# Import configuration
try:
    from .config import config
except ImportError:
    config = None


# Sensitive data patterns to mask
SENSITIVE_PATTERNS = [
    # API Keys - Fixed to match shorter keys too
    (r"sk-[A-Za-z0-9]{8,}", "sk-****"),
    (r"AIza[A-Za-z0-9\-_]{6,}", "AIza****"),
    (r"(api[_-]?key\s*[=:]\s*)([A-Za-z0-9\-_]+)", r"\1****"),
    (r"(Bearer\s+)([A-Za-z0-9\-_\.]+)", r"\1****"),
    # General tokens
    (r"(token\s*[=:]\s*)([A-Za-z0-9\-_]+)", r"\1****"),
    (r"(password\s*[=:]\s*)([^\s]+)", r"\1****"),
]


class SensitiveDataFilter(logging.Filter):
    """Filter that masks sensitive data in log messages."""

    def filter(self, record):
        # Mask sensitive data in the message
        if hasattr(record, "msg"):
            record.msg = mask_sensitive_data(str(record.msg))

        # Also mask in args if present
        if hasattr(record, "args") and record.args:
            record.args = tuple(mask_sensitive_data(str(arg)) for arg in record.args)

        return True


class ContextFilter(logging.Filter):
    """Filter that adds contextual information to log records."""

    def __init__(self):
        super().__init__()
        self.context = threading.local()

    def filter(self, record):
        # Add context data if available
        if hasattr(self.context, "data"):
            for key, value in self.context.data.items():
                setattr(record, key, value)
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
            ]:
                log_data[key] = value

        return json.dumps(log_data)


class AstrabotLogger(logging.Logger):
    """Enhanced logger with context management and utilities."""

    def __init__(self, name="astrabot"):
        super().__init__(name)
        self._context_filter = ContextFilter()
        self._sensitive_filter = SensitiveDataFilter()

        # Add filters to all handlers
        for handler in self.handlers:
            handler.addFilter(self._context_filter)
            handler.addFilter(self._sensitive_filter)

    @contextmanager
    def context(self, **kwargs):
        """Context manager for adding contextual information to logs."""
        # Store previous context
        previous = getattr(self._context_filter.context, "data", {}).copy()

        # Update context
        if not hasattr(self._context_filter.context, "data"):
            self._context_filter.context.data = {}

        self._context_filter.context.data.update(kwargs)

        try:
            yield self
        finally:
            # Restore previous context
            self._context_filter.context.data = previous


# Singleton logger instance
_logger_instance = None
_logger_lock = threading.Lock()


def get_logger(name: Optional[str] = None) -> AstrabotLogger:
    """
    Get the singleton logger instance.

    Args:
        name: Logger name (ignored, always returns singleton)

    Returns:
        AstrabotLogger instance
    """
    global _logger_instance

    # Fast path: return existing instance if available
    if _logger_instance is not None:
        return _logger_instance

    # Slow path: initialize the logger (with thread safety)
    with _logger_lock:
        # Double-check in case another thread initialized while we were waiting
        if _logger_instance is None:
            # Create logger
            logging.setLoggerClass(AstrabotLogger)
            logger = logging.getLogger("astrabot")

            # Ensure we have an AstrabotLogger instance
            if not isinstance(logger, AstrabotLogger):
                # Create a new AstrabotLogger instance with the same name
                logger = AstrabotLogger("astrabot")
                # Copy any existing handlers
                for handler in logging.getLogger("astrabot").handlers:
                    logger.addHandler(handler)

            _logger_instance = logger

            # Configure if not already configured
            if not _logger_instance.handlers:
                setup_logging()

    return _logger_instance


def mask_sensitive_data(text: str) -> str:
    """
    Mask sensitive data in text.

    Args:
        text: Text to mask

    Returns:
        Text with sensitive data masked
    """
    masked = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        masked = re.sub(pattern, replacement, masked, flags=re.IGNORECASE)
    return masked


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: str = "standard",
    stream: Optional[Any] = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
):
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (None for console only)
        log_format: 'standard' or 'json'
        stream: Stream to write to (default: sys.stdout)
        max_bytes: Max size for log file before rotation
        backup_count: Number of backup files to keep
    """
    logger = get_logger()

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set log level
    if log_level is None:
        log_level = os.environ.get("LOG_LEVEL", "INFO")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Determine format
    if log_format == "json" or os.environ.get("LOG_FORMAT") == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    # Console handler - default to stdout instead of stderr
    if stream is not None:
        console_handler = logging.StreamHandler(stream)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(logger._sensitive_filter)
    console_handler.addFilter(logger._context_filter)
    logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(logger._sensitive_filter)
        file_handler.addFilter(logger._context_filter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False


def log_performance(operation_name: str):
    """
    Decorator to log function performance.

    Args:
        operation_name: Name of the operation being timed
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(
                    f"{operation_name} completed in {elapsed:.2f}s",
                    extra={"operation": operation_name, "duration": elapsed},
                )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"{operation_name} failed after {elapsed:.2f}s: {str(e)}",
                    extra={"operation": operation_name, "duration": elapsed, "error": str(e)},
                )
                raise

        return wrapper

    return decorator


def log_api_call(api_name: str):
    """
    Decorator to log API calls.

    Args:
        api_name: Name of the API being called
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()

            # Log the call (but not sensitive args)
            safe_args = [mask_sensitive_data(str(arg)) for arg in args]
            safe_kwargs = {k: mask_sensitive_data(str(v)) for k, v in kwargs.items()}

            logger.info(
                f"Calling {api_name} API", extra={"api": api_name, "function": func.__name__}
            )

            start_time = time.time()
            try:
                result = func(*safe_args, **safe_kwargs)
                elapsed = time.time() - start_time
                logger.info(
                    f"{api_name} API call succeeded in {elapsed:.2f}s",
                    extra={"api": api_name, "duration": elapsed, "success": True},
                )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"{api_name} API call failed after {elapsed:.2f}s: {str(e)}",
                    extra={"api": api_name, "duration": elapsed, "success": False, "error": str(e)},
                )
                raise

        return wrapper

    return decorator


@contextmanager
def log_data_processing(data_type: str, count: Optional[int] = None):
    """
    Context manager for logging data processing operations.

    Args:
        data_type: Type of data being processed
        count: Number of items to process
    """
    logger = get_logger()
    start_time = time.time()

    context = {"data_type": data_type, "start_time": start_time, "processed": 0, "errors": 0}

    if count is not None:
        context["total"] = count

    logger.info(
        f"Starting {data_type} processing" + (f" ({count} items)" if count else ""),
        extra={"data_type": data_type, "count": count},
    )

    try:
        yield context
    finally:
        elapsed = time.time() - start_time
        logger.info(
            f"Completed {data_type} processing: {context['processed']} items in {elapsed:.2f}s",
            extra={
                "data_type": data_type,
                "processed": context["processed"],
                "errors": context["errors"],
                "duration": elapsed,
            },
        )


# Initialize logger with default settings when module is imported
_logger_instance = get_logger()
