"""The clinic's usage limits, which protect the free model quota and the real calendar."""

from dataclasses import dataclass

from a2a.server.agent_execution import RequestContext

from ragmed_core.ratelimit import RateLimiter

CLIENT_IP_HEADER = "x-ragmed-client-ip"
HOUR, DAY = 3600, 86400


@dataclass(frozen=True, slots=True)
class Limits:
    # A full intake (about ten questions, the report, times and a booking) is ~14 messages.
    messages_per_ip_per_hour: int = 40
    messages_per_conversation_per_day: int = 60
    # A ceiling on the whole demo, whoever sends: the free model quota is shared by everyone, so
    # a flood from many addresses ends in a friendly message instead of provider errors.
    messages_per_clinic_per_day: int = 1500
    bookings_per_ip_per_day: int = 3
    bookings_per_clinic_per_day: int = 40  # a ceiling on the whole calendar, whoever books


def client_ip(context: RequestContext) -> str:
    """The visitor's address, as the web app reported it. Only the web app can call this agent
    when service tokens are configured, so the header cannot be forged by a visitor."""
    headers = (context.call_context.state.get("headers") if context.call_context else None) or {}
    value = str(headers.get(CLIENT_IP_HEADER, "")).strip()
    return value[:64] or "unknown"


def wait_text(seconds: int) -> str:
    minutes = max(1, round(seconds / 60))
    if minutes < 90:  # noqa: PLR2004 - show minutes up to an hour and a half
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    return f"{round(minutes / 60)} hours"


async def message_limit(
    limiter: RateLimiter, limits: Limits, ip: str, conversation: str
) -> str | None:
    """None if this message may go ahead, else what to tell the visitor."""
    checks = [
        (f"msg:ip:{ip}", limits.messages_per_ip_per_hour, HOUR),
        (f"msg:conversation:{conversation}", limits.messages_per_conversation_per_day, DAY),
    ]
    for key, limit, window in checks:
        decision = await limiter.hit(key, limit, window)
        if not decision.allowed:
            return (
                "You have reached the message limit for now, which keeps this free demo "
                f"available for everyone. Please try again in {wait_text(decision.retry_after)}. "
                "If this is an emergency, call your local emergency number."
            )
    # Checked last, so visitors already over their own limit do not use up the shared budget.
    decision = await limiter.hit("msg:clinic", limits.messages_per_clinic_per_day, DAY)
    if not decision.allowed:
        return (
            "This free demo has reached its limit for today. Please come back tomorrow. "
            "If this is an emergency, call your local emergency number."
        )
    return None


async def booking_limit(limiter: RateLimiter, limits: Limits, ip: str) -> str | None:
    """None if a booking may be made, else what to tell the visitor."""
    if not (await limiter.hit(f"book:ip:{ip}", limits.bookings_per_ip_per_day, DAY)).allowed:
        return (
            "You have reached today's booking limit. Please call the clinic to book another visit."
        )
    if not (await limiter.hit("book:clinic", limits.bookings_per_clinic_per_day, DAY)).allowed:
        return "The clinic's online booking is full for today. Please call the clinic."
    return None
