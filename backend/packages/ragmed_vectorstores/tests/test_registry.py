from importlib.metadata import EntryPoint
from typing import Any

import pytest

from ragmed_vectorstores import registry
from ragmed_vectorstores.memory import InMemoryStore


def test_built_in_adapters_are_registered() -> None:
    assert {"memory", "qdrant"} <= set(registry.available_stores())


def test_load_store_passes_configuration() -> None:
    store = registry.load_store("memory", name="notes")

    assert isinstance(store, InMemoryStore)


def test_unknown_store_lists_the_available_ones() -> None:
    with pytest.raises(LookupError, match="memory"):
        registry.load_store("no-such-db")


def test_adapter_that_breaks_the_interface_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = EntryPoint("broken", "builtins:object", registry.GROUP)

    def only_broken(**kwargs: Any) -> list[EntryPoint]:
        return [fake] if kwargs.get("name") == "broken" else []

    monkeypatch.setattr(registry, "entry_points", only_broken)

    with pytest.raises(TypeError, match="VectorStore"):
        registry.load_store("broken")
