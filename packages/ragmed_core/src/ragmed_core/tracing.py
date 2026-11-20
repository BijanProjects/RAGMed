"""W3C Trace Context propagation.

One user turn gets one trace id. It travels across the Next.js proxy, every A2A hop
and every MCP call in the `traceparent` header, so the turn shows up as a single
trace in Langfuse and in the logs.
"""

import re
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import structlog

_TRACEPARENT = re.compile(r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")
_ZERO_TRACE = "0" * 32
_ZERO_SPAN = "0" * 16
_SAMPLED_FLAG = 0x01

_current: ContextVar["TraceContext | None"] = ContextVar("ragmed_trace", default=None)


@dataclass(frozen=True, slots=True)
class TraceContext:
    trace_id: str
    span_id: str
    sampled: bool = True

    @classmethod
    def new(cls) -> "TraceContext":
        return cls(trace_id=secrets.token_hex(16), span_id=secrets.token_hex(8))

    @classmethod
    def from_traceparent(cls, header: str | None) -> "TraceContext | None":
        """Parse a `traceparent` header. Returns None for anything malformed."""
        if not header:
            return None
        match = _TRACEPARENT.match(header.strip().lower())
        if match is None:
            return None
        version, trace_id, span_id, flags = match.groups()
        if version == "ff" or trace_id == _ZERO_TRACE or span_id == _ZERO_SPAN:
            return None
        return cls(trace_id=trace_id, span_id=span_id, sampled=bool(int(flags, 16) & _SAMPLED_FLAG))

    def child(self) -> "TraceContext":
        return TraceContext(self.trace_id, secrets.token_hex(8), self.sampled)

    def to_traceparent(self) -> str:
        flags = "01" if self.sampled else "00"
        return f"00-{self.trace_id}-{self.span_id}-{flags}"


def continue_or_start(header: str | None) -> TraceContext:
    """Continue the caller's trace with a new span, or start a fresh trace."""
    parent = TraceContext.from_traceparent(header)
    return parent.child() if parent else TraceContext.new()


def current_trace() -> TraceContext | None:
    return _current.get()


@contextmanager
def trace_scope(ctx: TraceContext) -> Iterator[TraceContext]:
    """Make `ctx` current and attach its ids to every log line inside the block."""
    token = _current.set(ctx)
    log_tokens = structlog.contextvars.bind_contextvars(trace_id=ctx.trace_id, span_id=ctx.span_id)
    try:
        yield ctx
    finally:
        structlog.contextvars.reset_contextvars(**log_tokens)
        _current.reset(token)
