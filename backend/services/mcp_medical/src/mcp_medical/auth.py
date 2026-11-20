"""Bearer tokens for the services allowed to call this server, one token per caller."""

import hmac
from collections.abc import Mapping

from fastmcp.server.auth import AccessToken, TokenVerifier

SCOPE = "knowledge:read"
MIN_TOKEN_LENGTH = 32


class ServiceTokenVerifier(TokenVerifier):
    """Accepts a token if it matches one of the configured callers' tokens.

    Every token is compared in constant time, so response timing reveals nothing about how
    close a guess was. The caller's name becomes the token's `client_id` for logs and traces.
    """

    def __init__(self, tokens: Mapping[str, str]) -> None:
        super().__init__(required_scopes=[SCOPE])
        if not tokens:
            raise ValueError("at least one service token is required")
        if any(len(token) < MIN_TOKEN_LENGTH for token in tokens.values()):
            raise ValueError(f"service tokens must be at least {MIN_TOKEN_LENGTH} characters")
        self._tokens = {caller: token.encode() for caller, token in tokens.items()}

    async def verify_token(self, token: str) -> AccessToken | None:
        presented = token.encode()
        caller = None
        for name, expected in self._tokens.items():
            if hmac.compare_digest(presented, expected):  # check all, to keep timing constant
                caller = name
        if caller is None:
            return None
        return AccessToken(token=token, client_id=caller, scopes=[SCOPE])
