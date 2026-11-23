"""Portable snapshots: a folder of Parquet files plus manifest.json.

    snapshot/
      manifest.json
      <collection>/part-00000.parquet ...

A snapshot is a backup you can keep anywhere, and it is also a Source, so `vf import`
is just a copy from a SnapshotSource into a sink (with resume and finalize for free),
and `vf verify` can check a DB against a snapshot.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import xxhash

from .connectors.base import Source
from .core.arrow import from_table, to_table
from .core.distance import distances
from .core.model import Batch, CollectionSpec, HnswParams, Metric, Record
from .engine.events import Emit, Event, noop

FORMAT = "vectorferry-snapshot/1"
ROWS_PER_PART = 50_000


def _safe_dir(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def export_snapshot(
    source: Source,
    out: str | Path,
    specs: list[CollectionSpec],
    *,
    batch_size: int = 500,
    emit: Emit = noop,
) -> Path:
    out = Path(out)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"{out} already holds a snapshot; pick an empty folder")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": FORMAT,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source.uri,
        "collections": [],
    }
    for spec in specs:
        cdir = out / _safe_dir(spec.name)
        cdir.mkdir(exist_ok=True)
        files, pending, cursor, done = [], [], None, 0
        emit(Event("collection_started", spec.name, 0, spec.count))

        def flush(
            cdir: Path = cdir,
            files: list = files,
            pending: list = pending,
            spec: CollectionSpec = spec,
        ) -> None:
            if not pending:
                return
            path = cdir / f"part-{len(files):05d}.parquet"
            pq.write_table(to_table(pending, spec.dim), path, compression="zstd")
            files.append(
                {
                    "path": str(path.relative_to(out)),
                    "rows": len(pending),
                    "xxh64": xxhash.xxh64_hexdigest(path.read_bytes()),
                }
            )
            pending.clear()

        while True:
            batch = source.scan(spec, cursor, batch_size)
            pending.extend(batch.records)
            done += len(batch.records)
            if len(pending) >= ROWS_PER_PART:
                flush()
            emit(Event("batch_done", spec.name, done, spec.count))
            cursor = batch.next_cursor
            if cursor is None:
                break
        flush()
        manifest["collections"].append(
            {
                "name": spec.name,
                "dim": spec.dim,
                "metric": spec.metric.value,
                "count": done,
                "hnsw": asdict(spec.hnsw),
                "files": files,
            }
        )
        emit(Event("collection_done", spec.name, done, spec.count))
    # Manifest last: a folder without one is an unfinished export.
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


class SnapshotSource(Source):
    def __init__(self, path: str | Path, *, check_files: bool = True) -> None:
        self.root = Path(path)
        mf = self.root / "manifest.json"
        if not mf.exists():
            raise FileNotFoundError(f"{self.root} has no manifest.json (unfinished export?)")
        self.manifest = json.loads(mf.read_text())
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"unsupported snapshot format {self.manifest.get('format')!r}")
        self.uri = f"snapshot://{self.root.resolve()}"
        if check_files:
            for c in self.manifest["collections"]:
                for f in c["files"]:
                    got = xxhash.xxh64_hexdigest((self.root / f["path"]).read_bytes())
                    if got != f["xxh64"]:
                        raise ValueError(f"{f['path']} is corrupt (checksum mismatch)")
        self._cache: dict[str, list[Record]] = {}

    def _entry(self, spec: CollectionSpec) -> dict:
        return next(c for c in self.manifest["collections"] if c["name"] == spec.name)

    def list_collections(self) -> list[CollectionSpec]:
        return [
            CollectionSpec(
                c["name"],
                c["dim"],
                Metric(c["metric"]),
                c["count"],
                hnsw=HnswParams(**c.get("hnsw", {})),
            )
            for c in self.manifest["collections"]
        ]

    def count(self, spec: CollectionSpec) -> int:
        return self._entry(spec)["count"]

    def scan(self, spec: CollectionSpec, cursor: str | None, batch_size: int) -> Batch:
        # cursor = "<part index>:<row offset>"
        files = self._entry(spec)["files"]
        part, offset = (int(x) for x in (cursor or "0:0").split(":"))
        if part >= len(files):
            return Batch([], None)
        table = pq.read_table(self.root / files[part]["path"])
        records = from_table(table.slice(offset, batch_size))
        offset += len(records)
        if offset >= table.num_rows:
            part, offset = part + 1, 0
        nxt = f"{part}:{offset}" if part < len(files) else None
        return Batch(records, nxt)

    def _all(self, spec: CollectionSpec) -> list[Record]:
        if spec.name not in self._cache:
            recs: list[Record] = []
            for f in self._entry(spec)["files"]:
                recs.extend(from_table(pq.read_table(self.root / f["path"])))
            self._cache[spec.name] = recs
        return self._cache[spec.name]

    def fetch(self, spec: CollectionSpec, ids: list[str]) -> list[Record]:
        want = set(ids)
        return [r for r in self._all(spec) if r.id in want]

    def query(
        self, spec: CollectionSpec, vector: list[float], k: int, *, exact: bool = True
    ) -> list[str]:
        """Always exact (brute force); a snapshot has no index."""
        recs = self._all(spec)
        m = np.asarray([r.vector for r in recs], dtype=np.float32)
        score = distances(spec.metric, np.asarray(vector, dtype=np.float32), m)
        top = np.argsort(score, kind="stable")[:k]
        return [recs[i].id for i in top]
