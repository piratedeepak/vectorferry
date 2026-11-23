"""Write to Postgres + pgvector.

Each batch is binary-COPYed into a temp staging table, then merged with
INSERT ... ON CONFLICT (id) DO UPDATE. That keeps COPY speed while making every batch
idempotent, which is what makes resume-after-crash and sync safe.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Jsonb

from ..core.hashing import record_hash
from ..core.model import CollectionSpec, Record
from ..mapping import IndexPlan, MappingError, check_record, index_plan, table_name
from .base import Sink

_STAGE = "_vf_stage"


class PgvectorSink(Sink):
    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "public",
        table_prefix: str = "vf_",
        table_map: dict[str, str] | None = None,
        no_index: bool = False,
    ) -> None:
        self.uri = _mask(dsn)
        self.schema, self.prefix, self.table_map, self.no_index = (
            schema,
            table_prefix,
            table_map,
            no_index,
        )
        self._conn = psycopg.connect(dsn, autocommit=False)
        with self._conn.transaction():
            have = self._conn.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            if not have:
                try:
                    self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                except psycopg.Error as e:
                    raise MappingError(
                        "pgvector is not installed in this database and this user cannot "
                        f"install it ({e.diag.message_primary}). Run CREATE EXTENSION vector "
                        "as a superuser first."
                    ) from e
        register_vector(self._conn)

    # -- helpers ------------------------------------------------------------------------------

    def table(self, spec: CollectionSpec) -> str:
        return table_name(spec.name, self.prefix, self.table_map)

    def target(self, spec: CollectionSpec) -> str:
        return f"{self.schema}.{self.table(spec)}"

    def _ident(self, spec: CollectionSpec) -> sql.Composed:
        return sql.Identifier(self.schema, self.table(spec))

    def plan(self, spec: CollectionSpec) -> IndexPlan:
        return index_plan(spec, no_index=self.no_index)

    # -- Sink ---------------------------------------------------------------------------------

    def ensure_collection(self, spec: CollectionSpec) -> None:
        t = self._ident(spec)
        with self._conn.transaction():
            existing = self._conn.execute(
                """SELECT format_type(a.atttypid, a.atttypmod)
                     FROM pg_attribute a
                    WHERE a.attrelid = to_regclass(%s) AND a.attname = 'embedding'""",
                (f'"{self.schema}"."{self.table(spec)}"',),
            ).fetchone()
            want = f"vector({spec.dim})"
            if existing and existing[0] != want:
                raise MappingError(
                    f"table {self.schema}.{self.table(spec)} already exists with embedding "
                    f"{existing[0]}, but collection {spec.name!r} needs {want}"
                )
            self._conn.execute(
                sql.SQL(
                    """CREATE TABLE IF NOT EXISTS {t} (
                         id        text PRIMARY KEY,
                         embedding vector({dim}) NOT NULL,
                         document  text,
                         metadata  jsonb NOT NULL DEFAULT '{{}}',
                         _vf_hash  bigint NOT NULL
                       )"""
                ).format(t=t, dim=sql.Literal(spec.dim))
            )
            self._conn.execute(
                sql.SQL("COMMENT ON TABLE {t} IS {c}").format(
                    t=t,
                    c=sql.Literal(
                        f"vectorferry: collection={spec.name} metric={spec.metric.value} "
                        f"dim={spec.dim}; for the source's search quality run "
                        f"SET hnsw.ef_search = {self.plan(spec).ef_search}"
                    ),
                )
            )

    def upsert(self, spec: CollectionSpec, records: list[Record]) -> None:
        if not records:
            return
        for r in records:
            check_record(r, spec.dim)
        t = self._ident(spec)
        with self._conn.transaction():
            self._conn.execute(
                f"""CREATE TEMP TABLE IF NOT EXISTS {_STAGE} (
                      id text, embedding vector, document text, metadata jsonb, _vf_hash bigint
                    ) ON COMMIT DELETE ROWS"""
            )
            with self._conn.cursor() as cur:
                copy_sql = (
                    f"COPY {_STAGE} (id, embedding, document, metadata, _vf_hash) "
                    "FROM STDIN WITH (FORMAT BINARY)"
                )
                with cur.copy(copy_sql) as copy:
                    copy.set_types(["text", "vector", "text", "jsonb", "int8"])
                    for r in records:
                        copy.write_row(
                            (
                                r.id,
                                np.asarray(r.vector, dtype=np.float32),
                                r.document,
                                Jsonb(r.metadata or {}),
                                record_hash(r),
                            )
                        )
            self._conn.execute(
                sql.SQL(
                    """INSERT INTO {t} (id, embedding, document, metadata, _vf_hash)
                       SELECT id, embedding, document, metadata, _vf_hash FROM {stage}
                       ON CONFLICT (id) DO UPDATE
                          SET embedding = EXCLUDED.embedding,
                              document  = EXCLUDED.document,
                              metadata  = EXCLUDED.metadata,
                              _vf_hash  = EXCLUDED._vf_hash
                        WHERE {t}._vf_hash IS DISTINCT FROM EXCLUDED._vf_hash"""
                ).format(t=t, stage=sql.Identifier(_STAGE))
            )

    def delete(self, spec: CollectionSpec, ids: list[str]) -> None:
        if not ids:
            return
        with self._conn.transaction():
            self._conn.execute(
                sql.SQL("DELETE FROM {t} WHERE id = ANY(%s)").format(t=self._ident(spec)), (ids,)
            )

    def finalize(self, spec: CollectionSpec) -> None:
        plan = self.plan(spec)
        t = self._ident(spec)
        with self._conn.transaction():
            if plan.kind == "hnsw":
                expr = (
                    sql.SQL("(embedding::halfvec({d}))").format(d=sql.Literal(spec.dim))
                    if plan.cast == "halfvec"
                    else sql.SQL("embedding")
                )
                # Build with the source's HNSW settings so search quality carries over.
                self._conn.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {ix} ON {t} USING hnsw ({expr} {ops}) "
                        "WITH (m = {m}, ef_construction = {efc})"
                    ).format(
                        ix=sql.Identifier(table_name(f"{self.table(spec)}_hnsw", prefix="")),
                        t=t,
                        expr=expr,
                        ops=sql.SQL(plan.opclass),
                        m=sql.Literal(plan.m),
                        efc=sql.Literal(plan.ef_construction),
                    )
                )
            self._conn.execute(sql.SQL("ANALYZE {t}").format(t=t))

    def count(self, spec: CollectionSpec) -> int:
        with self._conn.transaction():
            row = self._conn.execute(
                sql.SQL("SELECT count(*) FROM {t}").format(t=self._ident(spec))
            ).fetchone()
        return int(row[0])

    def fetch(self, spec: CollectionSpec, ids: list[str]) -> list[Record]:
        if not ids:
            return []
        with self._conn.transaction():
            rows = self._conn.execute(
                sql.SQL(
                    "SELECT id, embedding, document, metadata FROM {t} WHERE id = ANY(%s)"
                ).format(t=self._ident(spec)),
                (ids,),
            ).fetchall()
        return [
            Record(id=i, vector=_to_list(e), document=d, metadata=m or {}) for i, e, d, m in rows
        ]

    def query(
        self, spec: CollectionSpec, vector: list[float], k: int, *, exact: bool = False
    ) -> list[str]:
        plan = self.plan(spec)
        halfvec = plan.cast == "halfvec" and not exact
        expr = sql.SQL(plan.search_expr(spec.dim) if halfvec else "embedding")
        param = (
            sql.SQL("%s::halfvec({d})").format(d=sql.Literal(spec.dim))
            if halfvec
            else sql.SQL("%s::vector")
        )
        with self._conn.transaction():
            if exact:
                # Full-precision sequential scan: the ground truth the index is measured against.
                self._conn.execute("SET LOCAL enable_indexscan = off")
                self._conn.execute("SET LOCAL enable_bitmapscan = off")
            else:
                # Search with the source's ef_search (never below k, or HNSW returns < k rows).
                self._conn.execute(
                    sql.SQL("SET LOCAL hnsw.ef_search = {n}").format(
                        n=sql.Literal(min(max(plan.ef_search, k), 1000))
                    )
                )
            rows = self._conn.execute(
                sql.SQL("SELECT id FROM {t} ORDER BY {expr} {op} {param} LIMIT %s").format(
                    t=self._ident(spec), expr=expr, op=sql.SQL(plan.operator), param=param
                ),
                (np.asarray(vector, dtype=np.float32), k),
            ).fetchall()
        return [r[0] for r in rows]

    def stored_hashes(self, spec: CollectionSpec) -> dict[str, int]:
        """id -> stored checksum, for sync."""
        with self._conn.transaction():
            rows = self._conn.execute(
                sql.SQL("SELECT id, _vf_hash FROM {t}").format(t=self._ident(spec))
            ).fetchall()
        return dict(rows)

    def close(self) -> None:
        self._conn.close()


def _to_list(v: object) -> list[float]:
    # pgvector-python returns numpy arrays in older versions and Vector objects in newer ones.
    if hasattr(v, "to_list"):
        return v.to_list()
    return np.asarray(v).tolist()


def _mask(dsn: str) -> str:
    parts = urlsplit(dsn)
    if parts.password:
        netloc = parts.netloc.replace(f":{parts.password}@", ":***@")
        parts = parts._replace(netloc=netloc)
    return urlunsplit(parts._replace(query=""))
