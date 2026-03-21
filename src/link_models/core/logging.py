"""Structured logging configuration for link_models."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.traceback import install as install_rich_traceback

# Console for rich output
_console = Console(stderr=True)


def setup_logging(
    verbose: bool = False,
    json_format: bool = False,
    log_file: Path | None = None,
) -> None:
    """Configure structured logging for the application.
    
    Args:
        verbose: Enable debug level logging
        json_format: Output logs as JSON (for production/systemd)
        log_file: Optional file to write logs to
    """
    # Install rich traceback handler for better error display
    install_rich_traceback(console=_console, show_locals=verbose)
    
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    
    if json_format:
        # Production JSON format
        structlog.configure(
            processors=shared_processors + [
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                10 if verbose else 20  # DEBUG or INFO
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Development console format with rich colors
        structlog.configure(
            processors=shared_processors + [
                structlog.dev.ConsoleRenderer(
                    colors=True,
                    pad_level=False,
                ),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                10 if verbose else 20
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    # Add file handler if specified
    if log_file:
        # TODO: Implement file logging with rotation
        pass


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Optional logger name (uses module name if not provided)
        
    Returns:
        Configured structlog logger
    """
    if name is None:
        # Get caller's module name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            module = inspect.getmodule(frame.f_back)
            if module:
                name = module.__name__
        if name is None:
            name = "link_models"
    
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding structured context to logs."""
    
    def __init__(self, **context: Any) -> None:
        self.context = context
        self.token = None
    
    def __enter__(self) -> LogContext:
        self.token = structlog.contextvars.bind_contextvars(**self.context)
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        structlog.contextvars.reset_contextvars(self.token)


def log_action(
    logger: structlog.stdlib.BoundLogger,
    action: str,
    description: str,
    *,
    dry_run: bool = False,
    **kwargs: Any,
) -> None:
    """Log an action with dry-run support.
    
    Args:
        logger: Logger instance
        action: The action being performed (e.g., "link", "remove")
        description: Description of what's being done
        dry_run: Whether this is a dry-run
        **kwargs: Additional context to log
    """
    if dry_run:
        logger.info(
            f"[DRY-RUN] Would {action}: {description}",
            action=action,
            dry_run=True,
            **kwargs,
        )
    else:
        logger.info(
            f"{action.upper()}: {description}",
            action=action,
            **kwargs,
        )
