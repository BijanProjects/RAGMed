import logging
import re
import sys
from collections.abc import MutableMapping
from typing import Any, TextIO, cast

import structlog
from structlog.typing import FilteringBoundLogger

from ragmed_core.settings import CoreSettings

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)")

# Structural fields that must never be rewritten by redaction.
_SAFE_KEYS = frozenset({"event", "level", "timestamp", "logger", "service", "trace_id", "span_id"})


def redact(text: str) -> str:
    """Mask email addresses and phone numbers in free text."""
    text = _EMAIL.sub("[email]", text)
    return _PHONE.sub("[phone]", text)


def _redact_pii(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    for key, value in event_dict.items():
        if key not in _SAFE_KEYS and isinstance(value, str):
            event_dict[key] = redact(value)
    if isinstance(event_dict.get("event"), str):
        event_dict["event"] = redact(event_dict["event"])
    return event_dict


def configure_logging(settings: CoreSettings, stream: TextIO | None = None) -> None:
    """Configure structlog once per process. Safe to call again (e.g. in tests)."""
    level = logging.getLevelNamesMapping()[settings.log_level]
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if settings.use_json_logs
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact_pii,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stdout),
        context_class=dict,
        cache_logger_on_first_use=False,
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(service=settings.service_name)


def get_logger(name: str) -> FilteringBoundLogger:
    return cast(FilteringBoundLogger, structlog.get_logger().bind(logger=name))
