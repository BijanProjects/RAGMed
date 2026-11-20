"""Service-to-service bearer tokens: one per caller, compared in constant time."""

import hmac
from collections.abc import Mapping

MIN_TOKEN_LENGTH = 32


def check_tokens(tokens: Mapping[str, str]) -> None:
    """Refuse weak configuration: tokens must be long enough to resist guessing."""
    if any(len(token) < MIN_TOKEN_LENGTH for token in tokens.values()):
        raise ValueError(f"service tokens must be at least {MIN_TOKEN_LENGTH} characters")


def match_caller(presented: str, tokens: Mapping[str, str]) -> str | None:
    """The caller whose token matches, or None. Every token is compared, to keep timing even."""
    caller = None
    encoded = presented.encode()
    for name, expected in tokens.items():
        if hmac.compare_digest(encoded, expected.encode()):
            caller = name
    return caller
