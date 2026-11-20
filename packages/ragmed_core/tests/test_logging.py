import io
import json
from collections.abc import Iterator

import pytest
import structlog

from ragmed_core.logging import configure_logging, get_logger, redact
from ragmed_core.settings import CoreSettings
from ragmed_core.tracing import TraceContext, trace_scope


@pytest.fixture
def json_stream() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    settings = CoreSettings(_env_file=None, log_json=True, service_name="test-svc")
    configure_logging(settings, stream=stream)
    yield stream
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()


def _lines(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("contact jane.doe@example.com today", "contact [email] today"),
        ("call 709-555-0142 now", "call [phone] now"),
        ("call (709) 555 0142", "call [phone]"),
        ("dose is 500 mg twice daily", "dose is 500 mg twice daily"),
    ],
)
def test_redact(raw: str, expected: str) -> None:
    assert redact(raw) == expected


def test_json_log_line_carries_service_level_and_timestamp(json_stream: io.StringIO) -> None:
    get_logger("tests").info("retrieval_done", hits=5)

    [line] = _lines(json_stream)
    assert line["event"] == "retrieval_done"
    assert line["hits"] == 5
    assert line["level"] == "info"
    assert line["service"] == "test-svc"
    assert line["logger"] == "tests"
    assert "timestamp" in line


def test_pii_is_redacted_in_fields_and_event(json_stream: io.StringIO) -> None:
    get_logger("tests").info("booking for bob@example.org", attendee="bob@example.org")

    [line] = _lines(json_stream)
    assert line["event"] == "booking for [email]"
    assert line["attendee"] == "[email]"


def test_trace_ids_are_attached_inside_a_scope(json_stream: io.StringIO) -> None:
    ctx = TraceContext.new()
    log = get_logger("tests")

    with trace_scope(ctx):
        log.info("inside")
    log.info("outside")

    inside, outside = _lines(json_stream)
    assert inside["trace_id"] == ctx.trace_id
    assert inside["span_id"] == ctx.span_id
    assert "trace_id" not in outside


def test_level_filtering(json_stream: io.StringIO) -> None:
    get_logger("tests").debug("too_noisy")

    assert json_stream.getvalue() == ""


def test_console_renderer_for_local() -> None:
    stream = io.StringIO()
    configure_logging(CoreSettings(_env_file=None), stream=stream)
    try:
        get_logger("tests").info("hello")
    finally:
        structlog.reset_defaults()
        structlog.contextvars.clear_contextvars()

    output = stream.getvalue()
    assert "hello" in output
    assert not output.startswith("{")
