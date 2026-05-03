"""structlog setup.

We use structlog instead of stdlib logging because every line in this project
eventually ends up either (a) in MLflow as a tag/metric, or (b) in a CI log
that someone scrapes. Structured key=value output beats free-form strings
in both cases.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED = False


def configure(level: str = "INFO", *, json_logs: bool = False) -> None:
    """Configure structlog once per process.

    Calling this twice is a no-op — the second call would otherwise duplicate
    handlers and produce double-printed lines under pytest -s.

    Parameters
    ----------
    level
        Standard logging level name. Anything `logging` accepts.
    json_logs
        If True, render as line-delimited JSON. Used in Docker / CI where the
        log aggregator parses JSON. Default human-friendly key=value otherwise.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger. Configures lazily if needed."""
    if not _CONFIGURED:
        configure()
    return structlog.get_logger(name)  # type: ignore[no-any-return]
