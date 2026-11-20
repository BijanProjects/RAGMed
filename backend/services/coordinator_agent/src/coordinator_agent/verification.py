"""One-time codes that prove the patient owns the email a visit is booked under.

Without them, anyone could book visits, and send calendar invitations, in a stranger's name.
The code is emailed, never shown in the chat; only a salted hash is kept, in the conversation's
saved state. A code expires after ten minutes and allows five tries, and sending codes is
rate-limited per visitor and per address so the clinic cannot be used to flood an inbox.
"""

import hashlib
import hmac
import re
import secrets
import time
from typing import Any, TypedDict

from ragmed_core.ratelimit import RateLimiter

CODE_MINUTES = 10
MAX_ATTEMPTS = 5
CODES_PER_EMAIL_PER_HOUR = 3
CODES_PER_IP_PER_HOUR = 6
HOUR = 3600
EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")
CODE = re.compile(r"^\d{6}$")


class Pending(TypedDict):
    """A booking waiting for its code."""

    start: str
    clinician_type: str
    name: str
    email: str
    salt: str
    digest: str
    expires: float
    attempts: int


def _digest(salt: str, code: str) -> str:
    return hashlib.sha256(f"{salt}:{code}".encode()).hexdigest()


def new_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def pending_booking(
    code: str, *, start: str, clinician_type: str, name: str, email: str, now: float | None = None
) -> Pending:
    salt = secrets.token_hex(16)
    return Pending(
        start=start,
        clinician_type=clinician_type,
        name=name,
        email=email,
        salt=salt,
        digest=_digest(salt, code),
        expires=(now or time.time()) + CODE_MINUTES * 60,
        attempts=0,
    )


def code_matches(pending: Pending, code: str) -> bool:
    return CODE.match(code) is not None and hmac.compare_digest(
        pending["digest"], _digest(pending["salt"], code)
    )


def is_expired(pending: Pending, now: float | None = None) -> bool:
    return (now or time.time()) > pending["expires"]


async def send_limit(limiter: RateLimiter, ip: str, email: str) -> str | None:
    """None if another code may be sent, else what to tell the visitor."""
    address = hashlib.sha256(email.casefold().encode()).hexdigest()[:32]  # no addresses in Valkey
    if not (await limiter.hit(f"code:email:{address}", CODES_PER_EMAIL_PER_HOUR, HOUR)).allowed:
        return "Too many codes were sent to that address. Please try again in an hour."
    if not (await limiter.hit(f"code:ip:{ip}", CODES_PER_IP_PER_HOUR, HOUR)).allowed:
        return "Too many codes were requested. Please try again in an hour."
    return None


def public_view(pending: Pending | dict[str, Any]) -> dict[str, Any]:
    """What the web app may show: never the hash or salt."""
    return {
        "email": pending["email"],
        "start": pending["start"],
        "clinician_type": pending["clinician_type"],
        "minutes": CODE_MINUTES,
    }
