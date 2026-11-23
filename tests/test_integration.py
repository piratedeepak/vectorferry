"""End to end: every Chroma API generation -> pgvector, then verify, crash/resume, snapshot.

docker compose -f tests/compose.yml up -d
uv run pytest -m integration
"""

from __future__ import annotations

import pytest
from seed import seed

from vectorferry.connectors import open_sink, open_source
from vectorferry.engine.copy import copy_collection
from vectorferry.engine.state import State
from vectorferry.snapshot import SnapshotSource, export_snapshot
from vectorferry.verify import verify_collection

pytestmark = pytest.mark.integration

PG = "postgresql://vf:vf@localhost:15432/vf"
CHROMAS = {"0.4": 18004, "0.6": 18006, "1.0": 18100}
METRICS = ("cosine", "l2", "ip")


@pytest.fixture(scope="module", autouse=True)
def seeded():
    for port in CHROMAS.values():
        for m in METRICS:
            seed(f"http://localhost:{port}", f"it_{m}", 2000, 32, m)


@pytest.mark.parametrize("version", CHROMAS)
@pytest.mark.parametrize("metric", METRICS)
def test_copy_and_verify(tmp_path, version, metric):
    src = open_source(f"chroma://localhost:{CHROMAS[version]}")
    sink = open_sink(PG, table_prefix=f"it{version.replace('.', '')}_")
    spec = next(s for s in src.list_collections() if s.name == f"it_{metric}")
    assert spec.metric.value == metric  # incl. Chroma 0.6, whose config misreports it
    copy_collection(src, sink, spec, State(tmp_path / "s.db"), batch_size=300, restart=True)
    rep = verify_collection(src, sink, spec, sample=30, seed=0)
    assert rep.ok, rep


def test_resume_after_stop(tmp_path):
    src = open_source(f"chroma://localhost:{CHROMAS['1.0']}")
    sink = open_sink(PG, table_prefix="itresume_")
    spec = next(s for s in src.list_collections() if s.name == "it_cosine")
    st = State(tmp_path / "s.db")
    calls = iter(range(10_000))
    copied = copy_collection(
        src, sink, spec, st, batch_size=200, restart=True, should_stop=lambda: next(calls) >= 3
    )
    assert copied == 600  # stopped after 3 batches
    assert copy_collection(src, sink, spec, st, batch_size=200) == spec.count
    assert sink.count(spec) == spec.count
    assert verify_collection(src, sink, spec, sample=20, seed=0).ok


def test_snapshot_roundtrip(tmp_path):
    src = open_source(f"chroma://localhost:{CHROMAS['0.4']}")
    specs = [s for s in src.list_collections() if s.name.startswith("it_")]
    snap = SnapshotSource(export_snapshot(src, tmp_path / "snap", specs))
    sink = open_sink(PG, table_prefix="itsnap_")
    originals = {s.name: s for s in specs}
    for spec in snap.list_collections():
        assert spec.hnsw == originals[spec.name].hnsw  # index settings survive the snapshot
        copy_collection(snap, sink, spec, State(tmp_path / "s.db"), restart=True)
        # The restored table must match the ORIGINAL Chroma, not just the snapshot.
        assert verify_collection(src, sink, originals[spec.name], sample=20, seed=0).ok
