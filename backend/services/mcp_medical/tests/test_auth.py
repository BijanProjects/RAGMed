import pytest

from mcp_medical.auth import SCOPE, ServiceTokenVerifier

pytestmark = pytest.mark.anyio

DIAGNOSIS = "d" * 40
COORDINATOR = "c" * 40


async def test_a_known_token_identifies_its_caller() -> None:
    verifier = ServiceTokenVerifier({"diagnosis-agent": DIAGNOSIS, "coordinator": COORDINATOR})

    access = await verifier.verify_token(COORDINATOR)

    assert access is not None
    assert access.client_id == "coordinator"
    assert access.scopes == [SCOPE]


@pytest.mark.parametrize("token", ["", "wrong", "d" * 39, "d" * 41])
async def test_unknown_tokens_are_refused(token: str) -> None:
    verifier = ServiceTokenVerifier({"diagnosis-agent": DIAGNOSIS})

    assert await verifier.verify_token(token) is None


@pytest.mark.parametrize(
    ("tokens", "message"),
    [({}, "at least one"), ({"agent": "too-short"}, "32 characters")],
)
def test_weak_configuration_is_refused(tokens: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ServiceTokenVerifier(tokens)
