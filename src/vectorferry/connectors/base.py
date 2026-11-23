"""Connector interfaces. A DB that can be read is a Source; one that can be written is a Sink.

Phase 1 ships ChromaSource and PgvectorSink. Adding a DB means implementing one (or both)
of these in a new module; nothing else in the engine changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.model import Batch, CollectionSpec, Record


class Source(ABC):
    #: A display name with any password removed. Also the key for resume state.
    uri: str

    @abstractmethod
    def list_collections(self) -> list[CollectionSpec]: ...

    @abstractmethod
    def count(self, spec: CollectionSpec) -> int: ...

    @abstractmethod
    def scan(self, spec: CollectionSpec, cursor: str | None, batch_size: int) -> Batch:
        """Return one page of records. `cursor=None` starts from the beginning."""

    @abstractmethod
    def fetch(self, spec: CollectionSpec, ids: list[str]) -> list[Record]: ...

    @abstractmethod
    def query(self, spec: CollectionSpec, vector: list[float], k: int) -> list[str]:
        """IDs of the k nearest records, best first."""

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


class Sink(ABC):
    uri: str

    @abstractmethod
    def target(self, spec: CollectionSpec) -> str:
        """Where this collection lands, e.g. "public.vf_docs". Part of the resume key, so a
        different table/schema/prefix is a different job."""

    @abstractmethod
    def ensure_collection(self, spec: CollectionSpec) -> None:
        """Create the target collection/table if missing; fail if it exists but is incompatible."""

    @abstractmethod
    def upsert(self, spec: CollectionSpec, records: list[Record]) -> None: ...

    @abstractmethod
    def delete(self, spec: CollectionSpec, ids: list[str]) -> None: ...

    @abstractmethod
    def finalize(self, spec: CollectionSpec) -> None:
        """Called once after a full load, e.g. to build the vector index."""

    @abstractmethod
    def count(self, spec: CollectionSpec) -> int: ...

    @abstractmethod
    def fetch(self, spec: CollectionSpec, ids: list[str]) -> list[Record]: ...

    @abstractmethod
    def query(
        self, spec: CollectionSpec, vector: list[float], k: int, *, exact: bool = False
    ) -> list[str]:
        """IDs of the k nearest records, best first. exact=True bypasses the index."""

    def close(self) -> None:  # noqa: B027 - optional hook
        pass
