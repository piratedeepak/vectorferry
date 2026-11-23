"""How a source collection lands in pgvector: table name, index type, safety checks."""

from __future__ import annotations

from dataclasses import dataclass

import xxhash

from .core.model import CollectionSpec, Metric, Record

PG_IDENT_MAX = 63  # Postgres silently truncates longer identifiers, which can collide.
HNSW_VECTOR_MAX_DIM = 2000
HNSW_HALFVEC_MAX_DIM = 4000

_OPS = {
    Metric.COSINE: ("cosine_ops", "<=>"),
    Metric.L2: ("l2_ops", "<->"),
    Metric.IP: ("ip_ops", "<#>"),
}


class MappingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class IndexPlan:
    """How to index and search the embedding column.

    cast is None when the column is indexed as-is, "halfvec" when the index is an
    expression index on embedding::halfvec(dim) (dims 2001-4000), and kind is "none" when
    no HNSW index is possible or wanted (search then falls back to an exact scan).
    """

    kind: str  # "hnsw" | "none"
    opclass: str | None
    operator: str
    cast: str | None
    warning: str | None = None
    # pgvector's own limits: m 2..100, ef_construction >= 2*m (max 1000), ef_search 1..1000.
    m: int = 16
    ef_construction: int = 100
    ef_search: int = 100

    def search_expr(self, dim: int) -> str:
        return f"embedding::halfvec({dim})" if self.cast == "halfvec" else "embedding"


def index_plan(spec: CollectionSpec, *, no_index: bool = False) -> IndexPlan:
    ops, operator = _OPS[spec.metric]
    m = min(max(spec.hnsw.m, 2), 100)
    params = {
        "m": m,
        "ef_construction": min(max(spec.hnsw.ef_construction, 2 * m), 1000),
        "ef_search": min(max(spec.hnsw.ef_search, 1), 1000),
    }
    if no_index:
        return IndexPlan("none", None, operator, None, "index skipped (--no-index)", **params)
    if spec.dim <= HNSW_VECTOR_MAX_DIM:
        return IndexPlan("hnsw", f"vector_{ops}", operator, None, **params)
    if spec.dim <= HNSW_HALFVEC_MAX_DIM:
        return IndexPlan(
            "hnsw",
            f"halfvec_{ops}",
            operator,
            "halfvec",
            f"{spec.dim} dims is over pgvector's {HNSW_VECTOR_MAX_DIM}-dim HNSW limit for "
            "vector; indexing as halfvec (vectors are still stored at full precision)",
            **params,
        )
    return IndexPlan(
        "none",
        None,
        operator,
        None,
        f"{spec.dim} dims is over pgvector's {HNSW_HALFVEC_MAX_DIM}-dim HNSW limit; "
        "no index, searches will be exact scans",
        **params,
    )


def table_name(
    collection: str, prefix: str = "vf_", table_map: dict[str, str] | None = None
) -> str:
    if table_map and collection in table_map:
        name = table_map[collection]
        if len(name.encode()) > PG_IDENT_MAX:
            raise MappingError(f"table name {name!r} is longer than {PG_IDENT_MAX} bytes")
        return name
    name = f"{prefix}{collection}"
    if len(name.encode()) <= PG_IDENT_MAX:
        return name
    # Truncate but keep it unique: two long names sharing a prefix must not collide.
    suffix = "_" + xxhash.xxh32_hexdigest(collection.encode())
    head = name.encode()[: PG_IDENT_MAX - len(suffix)].decode(errors="ignore")
    return head + suffix


def check_record(rec: Record, dim: int) -> None:
    """Refuse records Postgres would reject or silently change, naming the record."""
    if len(rec.vector) != dim:
        raise MappingError(f"record {rec.id!r}: vector has {len(rec.vector)} dims, expected {dim}")
    if "\x00" in rec.id or (rec.document and "\x00" in rec.document):
        raise MappingError(f"record {rec.id!r}: contains a NUL byte, which Postgres text rejects")
    if rec.metadata and _has_nul(rec.metadata):
        raise MappingError(f"record {rec.id!r}: metadata contains a NUL byte, which jsonb rejects")


def _has_nul(value: object) -> bool:
    if isinstance(value, str):
        return "\x00" in value
    if isinstance(value, dict):
        return any(_has_nul(k) or _has_nul(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_has_nul(v) for v in value)
    return False
