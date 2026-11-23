"""Prove a copy is complete and correct.

1. Counts match.
2. Checksums: every source record exists in the sink with the same checksum. Checksums
   are recomputed from the data on both sides, never read from the sink's stored column,
   so corruption on the way in is caught.
3. Similarity measure: for sample queries, the sink's *exact* top-k is re-scored locally
   with the *source's* measure. It must come back in order, and no result the source
   returns may be closer than the sink's k-th exact result. A wrong mapping (say cosine
   stored as l2) fails this at once. It does not depend on how accurate the source's own
   ANN index is, which comparing two approximate indexes would.
4. Index recall: the sink's ANN index vs its own exact search. The question is "will
   search get worse after the move?", so it passes at >= --min-recall, OR when it is no
   worse than the source's own recall (within RECALL_SLACK). On hard data (e.g. random
   high-dimensional vectors) every HNSW, Chroma's included, misses most true neighbours;
   that is not a migration problem and must not fail a correct copy.

"Source agreement" is the source's own ANN recall measured against the same exact truth
(valid because the checksums prove both sides hold identical data).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .connectors.base import Sink, Source
from .core.distance import distances
from .core.hashing import record_hash
from .core.model import CollectionSpec, Record
from .engine.events import Emit, Event, noop

MAX_LISTED = 1000
REL_TOL = 1e-4  # float32 rounding and ties
RECALL_SLACK = 0.03  # sampling noise allowed when comparing to the source's own recall


@dataclass(slots=True)
class VerifyReport:
    collection: str
    source_count: int
    sink_count: int
    checked: int = 0
    missing: list[str] = field(default_factory=list)  # first MAX_LISTED ids
    mismatched: list[str] = field(default_factory=list)
    missing_total: int = 0
    mismatched_total: int = 0
    queries: int = 0
    metric_violations: int = 0  # queries where the sink's exact ranking contradicts the source
    index_recall: float | None = None  # sink ANN vs sink exact
    source_agreement: float | None = None  # source ANN vs sink exact (info only)
    min_recall: float = 0.95

    @property
    def counts_ok(self) -> bool:
        return self.source_count == self.sink_count

    @property
    def checksums_ok(self) -> bool:
        return not self.missing_total and not self.mismatched_total

    @property
    def metric_ok(self) -> bool:
        return self.metric_violations == 0

    @property
    def recall_ok(self) -> bool:
        if self.index_recall is None or self.index_recall >= self.min_recall:
            return True
        return (
            self.source_agreement is not None
            and self.index_recall >= self.source_agreement - RECALL_SLACK
        )

    @property
    def ok(self) -> bool:
        return self.counts_ok and self.checksums_ok and self.metric_ok and self.recall_ok


def verify_collection(
    source: Source,
    sink: Sink,
    spec: CollectionSpec,
    *,
    batch_size: int = 500,
    sample: int = 100,
    k: int = 10,
    min_recall: float = 0.95,
    checksums: bool = True,
    seed: int | None = None,
    emit: Emit = noop,
) -> VerifyReport:
    rep = VerifyReport(spec.name, source.count(spec), sink.count(spec), min_recall=min_recall)
    rng = random.Random(seed)
    # Reservoir-sample query vectors while scanning, so one pass serves every check.
    reservoir: list[Record] = []
    seen = 0

    cursor: str | None = None
    while True:
        batch = source.scan(spec, cursor, batch_size)
        for r in batch.records:
            seen += 1
            if len(reservoir) < sample:
                reservoir.append(r)
            elif (j := rng.randrange(seen)) < sample:
                reservoir[j] = r
        if checksums and batch.records:
            got = {r.id: r for r in sink.fetch(spec, [r.id for r in batch.records])}
            for r in batch.records:
                other = got.get(r.id)
                if other is None:
                    rep.missing_total += 1
                    _add(rep.missing, r.id)
                elif record_hash(other) != record_hash(r):
                    rep.mismatched_total += 1
                    _add(rep.mismatched, r.id)
            rep.checked += len(batch.records)
            emit(Event("batch_done", spec.name, rep.checked, rep.source_count))
        cursor = batch.next_cursor
        if cursor is None:
            break

    if reservoir and k > 0 and rep.sink_count:
        _search_checks(source, sink, spec, reservoir, min(k, rep.sink_count), rep)

    emit(
        Event(
            "collection_done",
            spec.name,
            rep.checked,
            rep.source_count,
            "ok" if rep.ok else "FAILED",
        )
    )
    return rep


def _search_checks(
    source: Source,
    sink: Sink,
    spec: CollectionSpec,
    queries: list[Record],
    k: int,
    rep: VerifyReport,
) -> None:
    index_hits = agree_hits = 0
    for r in queries:
        q = np.asarray(r.vector, dtype=np.float32)
        exact = sink.query(spec, r.vector, k, exact=True)
        ann = sink.query(spec, r.vector, k)
        src = source.query(spec, r.vector, k)
        index_hits += len(set(ann) & set(exact))
        agree_hits += len(set(src) & set(exact))

        # Re-score the sink's exact results (and the source's) with the source's measure.
        ids = list(dict.fromkeys(exact + src))
        vecs = {x.id: x.vector for x in sink.fetch(spec, ids)}
        known = [i for i in ids if i in vecs]
        d = dict(
            zip(
                known,
                distances(spec.metric, q, np.asarray([vecs[i] for i in known], np.float32)),
                strict=True,
            )
        )
        ex_d = [d[i] for i in exact if i in d]
        tol = REL_TOL * max(1.0, max(abs(x) for x in ex_d))
        out_of_order = any(b < a - tol for a, b in zip(ex_d, ex_d[1:], strict=False))
        kth = ex_d[-1]
        closer_than_kth = any(d[i] < kth - tol for i in src if i in d and i not in exact)
        if out_of_order or closer_than_kth:
            rep.metric_violations += 1

    n = len(queries) * k
    rep.queries = len(queries)
    rep.index_recall = index_hits / n
    rep.source_agreement = agree_hits / n


def _add(bucket: list[str], rid: str) -> None:
    if len(bucket) < MAX_LISTED:
        bucket.append(rid)
