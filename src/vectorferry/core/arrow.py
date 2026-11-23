"""Records <-> Arrow tables. Arrow/Parquet is the neutral format between DBs and on disk."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa

from .model import Record


def schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), dim), nullable=False),
            pa.field("document", pa.string()),
            # JSON text: keeps any metadata shape without a per-collection schema.
            pa.field("metadata", pa.string()),
        ]
    )


def to_table(records: list[Record], dim: int) -> pa.Table:
    vectors = np.asarray([r.vector for r in records], dtype=np.float32).reshape(len(records), dim)
    return pa.table(
        {
            "id": pa.array([r.id for r in records], pa.string()),
            "vector": pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), dim),
            "document": pa.array([r.document for r in records], pa.string()),
            "metadata": pa.array(
                [json.dumps(r.metadata, sort_keys=True) if r.metadata else None for r in records],
                pa.string(),
            ),
        },
        schema=schema(dim),
    )


def from_table(table: pa.Table) -> list[Record]:
    ids = table.column("id").to_pylist()
    dim = table.schema.field("vector").type.list_size
    flat = table.column("vector").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    vectors = flat.reshape(len(ids), dim)
    docs = table.column("document").to_pylist()
    metas = table.column("metadata").to_pylist()
    return [
        Record(id=i, vector=v.tolist(), document=d, metadata=json.loads(m) if m else {})
        for i, v, d, m in zip(ids, vectors, docs, metas, strict=True)
    ]
