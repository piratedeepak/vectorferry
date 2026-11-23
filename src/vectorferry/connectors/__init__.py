"""Open a Source or Sink from a URI.

Sources:
  chroma://host:port            Chroma server over HTTP (any version, 0.4 -> 1.x)
  chroma+https://host[:port]    same, over TLS
  snapshot:///path/to/folder    a folder written by `vf export`  (a bare existing path works too)
Sinks:
  postgresql://user:pass@host:port/db   Postgres with pgvector
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from .base import Sink, Source


def open_source(
    uri: str, *, tenant: str = "default_tenant", database: str = "default_database"
) -> Source:
    parts = urlsplit(uri)
    if parts.scheme in ("chroma", "chroma+http", "chroma+https"):
        from .chroma import ChromaSource

        scheme = "https" if parts.scheme == "chroma+https" else "http"
        port = parts.port or (443 if scheme == "https" else 8000)
        return ChromaSource(
            f"{scheme}://{parts.hostname}:{port}",
            tenant=tenant,
            database=database,
            token=os.environ.get("VF_CHROMA_TOKEN"),
        )
    if parts.scheme == "snapshot" or (not parts.scheme and Path(uri).is_dir()):
        from ..snapshot import SnapshotSource

        return SnapshotSource(parts.path if parts.scheme else uri)
    raise ValueError(f"don't know how to read {uri!r} (expected chroma://host:port or a snapshot)")


def open_sink(uri: str, **kw) -> Sink:
    if urlsplit(uri).scheme in ("postgres", "postgresql"):
        from .pgvector import PgvectorSink

        return PgvectorSink(uri, **kw)
    raise ValueError(f"don't know how to write {uri!r} (expected postgresql://...)")
