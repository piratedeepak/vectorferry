"""Read from a running Chroma server over its HTTP API.

This talks HTTP directly instead of using the `chromadb` package on purpose: a chromadb
1.x client cannot talk to a 0.4 server (and the other way round), but the HTTP routes are
simple and stable enough to support both families from one tool:

* /api/v1/collections/...                                         (0.4.x, 0.5.x)
* /api/v2/tenants/{tenant}/databases/{database}/collections/...   (0.6+, 1.x)
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import httpx

from ..core.model import Batch, CollectionSpec, HnswParams, Metric, Record
from .base import Source

_INCLUDE = ["embeddings", "documents", "metadatas"]


class ChromaError(RuntimeError):
    pass


class ChromaSource(Source):
    def __init__(
        self,
        base_url: str,
        *,
        tenant: str = "default_tenant",
        database: str = "default_database",
        token: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)
        parts = urlsplit(base_url)
        self.uri = f"chroma://{parts.hostname}:{parts.port or 8000}/{tenant}/{database}"
        self.tenant, self.database = tenant, database
        self.api = self._detect_api()

    # -- plumbing -----------------------------------------------------------------------------

    def _detect_api(self) -> str:
        for api in ("v2", "v1"):
            try:
                r = self._http.get(f"/api/{api}/heartbeat")
            except httpx.HTTPError as e:
                raise ChromaError(f"cannot reach Chroma at {self._http.base_url}: {e}") from e
            if r.status_code == 200:
                return api
        raise ChromaError(f"{self._http.base_url} does not look like a Chroma server")

    def _collections_path(self) -> str:
        if self.api == "v2":
            return f"/api/v2/tenants/{self.tenant}/databases/{self.database}/collections"
        return "/api/v1/collections"

    def _call(self, method: str, path: str, **kw: Any) -> Any:
        r = self._http.request(method, path, **kw)
        if r.status_code >= 400:
            raise ChromaError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def _col(self, spec: CollectionSpec) -> str:
        return f"{self._collections_path()}/{spec.extra['chroma_id']}"

    # -- Source -------------------------------------------------------------------------------

    def list_collections(self) -> list[CollectionSpec]:
        out = []
        for c in self._call("GET", self._collections_path()):
            spec = CollectionSpec(
                name=c["name"],
                dim=c.get("dimension") or 0,
                metric=_metric_of(c),
                hnsw=_hnsw_of(c),
                extra={"chroma_id": c["id"]},
            )
            spec.count = self.count(spec)
            if not spec.dim and spec.count:
                # 0.x servers never report a dimension; read it off the first record.
                first = self.scan(spec, None, 1).records
                spec.dim = len(first[0].vector) if first else 0
            out.append(spec)
        return out

    def count(self, spec: CollectionSpec) -> int:
        return int(self._call("GET", f"{self._col(spec)}/count"))

    def scan(self, spec: CollectionSpec, cursor: str | None, batch_size: int) -> Batch:
        offset = int(cursor or 0)
        body = {"limit": batch_size, "offset": offset, "include": _INCLUDE}
        records = _records(self._call("POST", f"{self._col(spec)}/get", json=body))
        nxt = str(offset + len(records)) if len(records) == batch_size else None
        return Batch(records=records, next_cursor=nxt)

    def fetch(self, spec: CollectionSpec, ids: list[str]) -> list[Record]:
        if not ids:
            return []
        body = {"ids": ids, "include": _INCLUDE}
        return _records(self._call("POST", f"{self._col(spec)}/get", json=body))

    def query(self, spec: CollectionSpec, vector: list[float], k: int) -> list[str]:
        body = {"query_embeddings": [vector], "n_results": k, "include": ["distances"]}
        return self._call("POST", f"{self._col(spec)}/query", json=body)["ids"][0]

    def close(self) -> None:
        self._http.close()


def _hnsw_of(c: dict[str, Any]) -> HnswParams:
    """Same precedence as the measure: metadata (0.4-0.6 and 1.x when set), then 1.x config."""
    meta = c.get("metadata") or {}
    cfg = (c.get("configuration_json") or {}).get("hnsw") or {}
    p = HnswParams()
    p.m = int(meta.get("hnsw:M") or cfg.get("max_neighbors") or p.m)
    p.ef_construction = int(
        meta.get("hnsw:construction_ef") or cfg.get("ef_construction") or p.ef_construction
    )
    p.ef_search = int(meta.get("hnsw:search_ef") or cfg.get("ef_search") or p.ef_search)
    return p


def _metric_of(c: dict[str, Any]) -> Metric:
    # Metadata wins. Chroma 0.6 reports configuration_json.hnsw_configuration.space = "l2"
    # as a placeholder even when the collection was created with hnsw:space = "ip"; the
    # metadata is what it actually indexes with. 1.x moved the setting into
    # configuration_json.hnsw / .spann, which only matters when metadata has no hnsw:space.
    space = (c.get("metadata") or {}).get("hnsw:space")
    if space:
        return Metric(space)
    cfg = c.get("configuration_json") or {}
    for index in ("hnsw", "spann"):
        space = (cfg.get(index) or {}).get("space")
        if space:
            return Metric(space)
    return Metric.L2  # Chroma's default


def _records(payload: dict[str, Any]) -> list[Record]:
    ids = payload["ids"]
    embs = payload.get("embeddings") or [None] * len(ids)
    docs = payload.get("documents") or [None] * len(ids)
    metas = payload.get("metadatas") or [None] * len(ids)
    out = []
    for i, e, d, m in zip(ids, embs, docs, metas, strict=True):
        if e is None:
            raise ChromaError(f"record {i!r} has no embedding")
        out.append(Record(id=i, vector=list(e), document=d, metadata=m or {}))
    return out
