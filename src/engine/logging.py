"""Structured logging setup. Call ``configure_logging`` once at process start
(the CLI does this); libraries just do ``structlog.get_logger(__name__)``.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(*, json_output: bool = False, level: int = logging.INFO) -> None:
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        # Resolve the output stream at emit time and never cache the bound
        # logger: a cached logger holds whatever sys.stderr was at configure
        # time, which breaks under test runners that swap/close the stream.
        logger_factory=lambda *args: structlog.PrintLogger(sys.stderr),
        cache_logger_on_first_use=False,
    )
