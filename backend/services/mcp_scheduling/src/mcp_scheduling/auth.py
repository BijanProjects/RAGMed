"""Bearer tokens for the services allowed to book (the Care Coordinator)."""

from collections.abc import Mapping

from fastmcp.server.auth import AccessToken, TokenVerifier

from ragmed_core.tokens import check_tokens, match_caller

SCOPE = "scheduling"


class ServiceTokenVerifier(TokenVerifier):
    def __init__(self, tokens: Mapping[str, str]) -> None:
        super().__init__(required_scopes=[SCOPE])
        if not tokens:
            raise ValueError("at least one service token is required")
        check_tokens(tokens)
        self._tokens = dict(tokens)

    async def verify_token(self, token: str) -> AccessToken | None:
        caller = match_caller(token, self._tokens)
        if caller is None:
            return None
        return AccessToken(token=token, client_id=caller, scopes=[SCOPE])
