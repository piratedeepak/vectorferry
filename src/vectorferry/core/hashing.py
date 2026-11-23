"""A stable per-record checksum, the same whichever DB the record came from.

Vectors are hashed as little-endian float32 so a float64 round-trip through one DB
and float32 storage in another still match. Metadata is hashed as canonical JSON
(sorted keys, no whitespace) so key order never matters.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import xxhash

from .model import Record

_SEP = b"\x1f"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def record_hash(record: Record) -> int:
    """Signed 64-bit hash, so it fits a Postgres bigint."""
    h = xxhash.xxh64()
    h.update(record.id.encode())
    h.update(_SEP)
    h.update(np.asarray(record.vector, dtype="<f4").tobytes())
    h.update(_SEP)
    h.update(canonical_json(record.metadata or {}))
    h.update(_SEP)
    if record.document is not None:
        h.update(b"\x01")
        h.update(record.document.encode())
    else:
        h.update(b"\x00")
    unsigned = h.intdigest()
    return unsigned - (1 << 64) if unsigned >= (1 << 63) else unsigned
