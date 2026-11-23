"""The DB-neutral data model every connector reads and writes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Metric(StrEnum):
    """Similarity measure a collection was indexed with."""

    COSINE = "cosine"
    L2 = "l2"
    IP = "ip"  # inner product / dot product


@dataclass(slots=True)
class Record:
    id: str
    vector: list[float]
    document: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HnswParams:
    """The source's HNSW settings, carried over so search quality matches after the move.
    Defaults are Chroma's."""

    m: int = 16
    ef_construction: int = 100
    ef_search: int = 100


@dataclass(slots=True)
class CollectionSpec:
    name: str
    dim: int
    metric: Metric
    count: int | None = None
    hnsw: HnswParams = field(default_factory=HnswParams)
    # Anything source-specific worth keeping (e.g. the Chroma collection id).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Batch:
    """A page of records plus the cursor to ask for the next page (None = done)."""

    records: list[Record]
    next_cursor: str | None
