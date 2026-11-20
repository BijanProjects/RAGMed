import os

import pytest


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests independent of RAGMED_* variables set by the shell or CI."""
    for name in list(os.environ):
        if name.startswith("RAGMED_"):
            monkeypatch.delenv(name)
