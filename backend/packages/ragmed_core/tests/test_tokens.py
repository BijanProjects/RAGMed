import pytest

from ragmed_core.tokens import check_tokens, match_caller

TOKENS = {"diagnosis-agent": "d" * 40, "coordinator": "c" * 40}


def test_the_matching_caller_is_returned() -> None:
    assert match_caller("c" * 40, TOKENS) == "coordinator"


@pytest.mark.parametrize("presented", ["", "c" * 39, "x" * 40])
def test_unknown_tokens_match_nobody(presented: str) -> None:
    assert match_caller(presented, TOKENS) is None


def test_short_tokens_are_refused() -> None:
    with pytest.raises(ValueError, match="32"):
        check_tokens({"weak": "short"})
