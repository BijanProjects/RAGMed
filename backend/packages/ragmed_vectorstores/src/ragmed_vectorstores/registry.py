"""Find vector store adapters by name, including ones installed by other packages."""

from importlib.metadata import entry_points
from typing import Any

from ragmed_vectorstores.base import VectorStore

GROUP = "ragmed.vectorstores"


def available_stores() -> list[str]:
    return sorted(ep.name for ep in entry_points(group=GROUP))


def load_store(adapter: str, /, **config: Any) -> VectorStore:
    """Create the adapter registered as `adapter`, e.g. load_store("qdrant", url=...).

    `config` goes to the adapter's constructor unchanged.
    """
    matches = entry_points(group=GROUP, name=adapter)
    if not matches:
        raise LookupError(f"no vector store named {adapter!r}; available: {available_stores()}")
    factory = next(iter(matches)).load()
    store = factory(**config)
    if not isinstance(store, VectorStore):
        raise TypeError(f"{adapter!r} does not implement the VectorStore interface")
    return store
