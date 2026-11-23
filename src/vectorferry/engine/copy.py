"""Copy collections from a Source to a Sink in batches, resumable after a crash."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from ..connectors.base import Sink, Source
from ..core.model import CollectionSpec
from .events import Emit, Event, noop
from .state import State

T = TypeVar("T")


def with_retries(
    fn: Callable[[], T],
    *,
    what: str,
    collection: str,
    emit: Emit,
    attempts: int = 4,
    base_delay: float = 1.0,
) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (OSError, ConnectionError, TimeoutError) as e:
            retryable = e
        except Exception as e:  # httpx / psycopg transient errors without importing both here
            name = type(e).__name__
            if not any(s in name for s in ("Timeout", "Connect", "Operational", "Network")):
                raise
            retryable = e
        if attempt == attempts:
            raise retryable
        delay = base_delay * 2 ** (attempt - 1)
        emit(
            Event(
                "retry", collection, message=f"{what} failed ({retryable}); retry in {delay:.0f}s"
            )
        )
        time.sleep(delay)
    raise AssertionError("unreachable")


def copy_collection(
    source: Source,
    sink: Sink,
    spec: CollectionSpec,
    state: State,
    *,
    batch_size: int = 500,
    restart: bool = False,
    emit: Emit = noop,
    should_stop: Callable[[], bool] = lambda: False,
) -> int:
    """Copy one collection. Returns the number of records copied in total (incl. earlier runs)."""
    job = f"copy:{source.uri}->{sink.uri}/{sink.target(spec)}"
    if restart:
        state.reset(job, spec.name)
    prog = state.get(job, spec.name)
    sink.ensure_collection(spec)
    if prog and sink.count(spec) < prog.copied:
        # The target lost rows since the last run (dropped, truncated): saved progress
        # can't be trusted, so start this collection over. Upserts make that safe.
        emit(
            Event(
                "warning",
                spec.name,
                0,
                spec.count,
                f"target has fewer rows than the {prog.copied} recorded as copied; "
                "starting this collection over",
            )
        )
        state.reset(job, spec.name)
        prog = None
    if prog and prog.done:
        emit(Event("collection_done", spec.name, prog.copied, prog.copied, "already copied"))
        return prog.copied

    cursor, copied = (prog.cursor, prog.copied) if prog else (None, 0)
    total = spec.count if spec.count is not None else source.count(spec)
    emit(Event("collection_started", spec.name, copied, total, "resuming" if prog else None))

    while True:
        if should_stop():
            emit(Event("warning", spec.name, copied, total, "paused"))
            return copied
        batch = with_retries(
            lambda c=cursor: source.scan(spec, c, batch_size),
            what="read",
            collection=spec.name,
            emit=emit,
        )
        if batch.records:
            with_retries(
                lambda b=batch: sink.upsert(spec, b.records),
                what="write",
                collection=spec.name,
                emit=emit,
            )
            copied += len(batch.records)
        cursor = batch.next_cursor
        state.save(job, spec.name, cursor, copied, done=False)
        emit(Event("batch_done", spec.name, copied, total))
        if cursor is None:
            break

    sink.finalize(spec)
    state.save(job, spec.name, None, copied, done=True)
    emit(Event("collection_done", spec.name, copied, total))
    return copied
