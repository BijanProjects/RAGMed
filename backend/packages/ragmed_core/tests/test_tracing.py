import re

import pytest

from ragmed_core.tracing import TraceContext, continue_or_start, current_trace, trace_scope

VALID = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def test_new_context_has_spec_compliant_ids() -> None:
    ctx = TraceContext.new()

    assert re.fullmatch(r"[0-9a-f]{32}", ctx.trace_id)
    assert re.fullmatch(r"[0-9a-f]{16}", ctx.span_id)
    assert ctx.sampled is True


def test_parse_valid_header() -> None:
    ctx = TraceContext.from_traceparent(VALID)

    assert ctx == TraceContext(
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736", span_id="00f067aa0ba902b7", sampled=True
    )


def test_round_trip() -> None:
    ctx = TraceContext.new()

    assert TraceContext.from_traceparent(ctx.to_traceparent()) == ctx


def test_unsampled_flag_is_preserved() -> None:
    ctx = TraceContext.from_traceparent(VALID[:-2] + "00")

    assert ctx is not None
    assert ctx.sampled is False
    assert ctx.to_traceparent().endswith("-00")


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        "garbage",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
        "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
        "00-4bf92f3577b34da6a3ce929d0e0e47zz-00f067aa0ba902b7-01",
    ],
)
def test_malformed_headers_are_rejected(header: str | None) -> None:
    assert TraceContext.from_traceparent(header) is None


def test_continue_keeps_trace_id_with_new_span() -> None:
    ctx = continue_or_start(VALID)

    assert ctx.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert ctx.span_id != "00f067aa0ba902b7"


def test_continue_starts_fresh_trace_without_header() -> None:
    first, second = continue_or_start(None), continue_or_start(None)

    assert first.trace_id != second.trace_id


def test_scope_sets_and_restores_current_trace() -> None:
    outer, inner = TraceContext.new(), TraceContext.new()

    assert current_trace() is None
    with trace_scope(outer):
        assert current_trace() == outer
        with trace_scope(inner):
            assert current_trace() == inner
        assert current_trace() == outer
    assert current_trace() is None
