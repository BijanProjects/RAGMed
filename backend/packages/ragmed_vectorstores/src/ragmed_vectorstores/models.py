"""The data every vector store adapter accepts and returns."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

MetadataValue = str | int | float | bool
Metadata = Mapping[str, MetadataValue]
# Filters match exactly, which is unreliable for floats, so they are not allowed in filters.
FilterValue = str | int | bool


def check_filter(filter_: Mapping[str, FilterValue] | None) -> None:
    for key, value in (filter_ or {}).items():
        if isinstance(value, float):
            raise TypeError(f"filter on {key!r}: float values are not supported")


@dataclass(frozen=True, slots=True)
class Document:
    """A passage to store: its text, its embedding, and metadata to filter on."""

    id: str
    text: str
    vector: Sequence[float]
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Query:
    """Find the `top_k` documents closest to `vector`.

    `filter` keeps only documents whose metadata equals every given value.
    """

    vector: Sequence[float]
    top_k: int = 5
    filter: Mapping[str, FilterValue] | None = None


@dataclass(frozen=True, slots=True)
class Hit:
    id: str
    score: float
    text: str
    metadata: Metadata


@dataclass(frozen=True, slots=True)
class Capabilities:
    hybrid: bool
    filters: bool
    max_dimensions: int | None = None


@dataclass(frozen=True, slots=True)
class Health:
    ok: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    name: str
    dimensions: int | None
    count: int
    metric: Literal["cosine"] = "cosine"


@dataclass(frozen=True, slots=True)
class UpsertResult:
    count: int
