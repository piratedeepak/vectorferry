import numpy as np
import pytest

from vectorferry.connectors.chroma import _metric_of
from vectorferry.core.arrow import from_table, to_table
from vectorferry.core.distance import distances
from vectorferry.core.hashing import record_hash
from vectorferry.core.model import CollectionSpec, Metric, Record
from vectorferry.mapping import MappingError, check_record, index_plan, table_name
from vectorferry.snapshot import SnapshotSource, export_snapshot
from vectorferry.verify import verify_collection


def rec(i: str, v=(0.1, 0.2, 0.3), doc="d", meta=None) -> Record:
    return Record(i, list(v), doc, meta if meta is not None else {"k": 1, "tag": "x"})


# -- hashing -------------------------------------------------------------------------------------


def test_hash_ignores_metadata_key_order():
    a = rec("a", meta={"x": 1, "y": "two"})
    b = rec("a", meta={"y": "two", "x": 1})
    assert record_hash(a) == record_hash(b)


def test_hash_float64_vs_float32_roundtrip_match():
    v64 = [0.1, 0.2, 0.30000000000000004]
    v32 = np.asarray(v64, dtype=np.float32).tolist()
    assert record_hash(rec("a", v64)) == record_hash(rec("a", v32))


@pytest.mark.parametrize(
    "change",
    [
        lambda r: setattr(r, "id", "b"),
        lambda r: setattr(r, "document", None),
        lambda r: setattr(r, "document", ""),
        lambda r: r.vector.__setitem__(0, 0.5),
        lambda r: r.metadata.__setitem__("k", 2),
    ],
)
def test_hash_changes_with_any_field(change):
    r = rec("a")
    before = record_hash(r)
    change(r)
    assert record_hash(r) != before


def test_hash_fits_bigint():
    for i in range(500):
        h = record_hash(rec(str(i)))
        assert -(1 << 63) <= h < (1 << 63)


# -- arrow ---------------------------------------------------------------------------------------


def test_arrow_roundtrip_keeps_everything():
    recs = [
        rec("a"),
        Record("b", [1.0, 2.0, 3.0], None, {}),
        rec("c", meta={"n": {"deep": [1, 2]}}),
    ]
    back = from_table(to_table(recs, 3))
    assert [record_hash(r) for r in back] == [record_hash(r) for r in recs]


# -- mapping -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric,opclass,op",
    [
        (Metric.COSINE, "vector_cosine_ops", "<=>"),
        (Metric.L2, "vector_l2_ops", "<->"),
        (Metric.IP, "vector_ip_ops", "<#>"),
    ],
)
def test_index_plan_per_metric(metric, opclass, op):
    p = index_plan(CollectionSpec("c", 768, metric))
    assert (p.kind, p.opclass, p.operator, p.cast) == ("hnsw", opclass, op, None)


def test_index_plan_halfvec_and_none_by_dim():
    assert index_plan(CollectionSpec("c", 2000, Metric.COSINE)).cast is None
    p = index_plan(CollectionSpec("c", 3072, Metric.COSINE))
    assert (p.opclass, p.cast) == ("halfvec_cosine_ops", "halfvec") and p.warning
    p = index_plan(CollectionSpec("c", 4096, Metric.COSINE))
    assert p.kind == "none" and p.warning


def test_table_name_long_names_stay_unique_and_short():
    a = table_name("x" * 80 + "a")
    b = table_name("x" * 80 + "b")
    assert a != b and len(a.encode()) <= 63 and len(b.encode()) <= 63
    assert table_name("docs") == "vf_docs"
    assert table_name("docs", table_map={"docs": "my_docs"}) == "my_docs"


def test_check_record_rejects_what_postgres_would():
    check_record(rec("ok"), 3)
    with pytest.raises(MappingError, match="dims"):
        check_record(rec("a"), 4)
    with pytest.raises(MappingError, match="NUL"):
        check_record(rec("a", doc="bad\x00doc"), 3)
    with pytest.raises(MappingError, match="metadata"):
        check_record(rec("a", meta={"k": ["fine", "bad\x00"]}), 3)


# -- chroma metric detection ---------------------------------------------------------------------


def test_metric_metadata_beats_placeholder_config():
    # Chroma 0.6 reports this placeholder "l2" config for a collection created with ip.
    c = {
        "metadata": {"hnsw:space": "ip"},
        "configuration_json": {"hnsw_configuration": {"space": "l2"}},
    }
    assert _metric_of(c) == Metric.IP


def test_hnsw_settings_carry_over_with_metadata_first():
    from vectorferry.connectors.chroma import _hnsw_of

    p = _hnsw_of({"metadata": {"hnsw:M": 32, "hnsw:construction_ef": 200, "hnsw:search_ef": 10}})
    assert (p.m, p.ef_construction, p.ef_search) == (32, 200, 10)
    p = _hnsw_of(
        {
            "configuration_json": {
                "hnsw": {"max_neighbors": 24, "ef_construction": 150, "ef_search": 80}
            }
        }
    )
    assert (p.m, p.ef_construction, p.ef_search) == (24, 150, 80)
    p = _hnsw_of({})
    assert (p.m, p.ef_construction, p.ef_search) == (16, 100, 100)
    # pgvector limits are respected: ef_construction >= 2*m, everything clamped.
    spec = CollectionSpec("c", 8, Metric.L2)
    spec.hnsw.m, spec.hnsw.ef_construction, spec.hnsw.ef_search = 200, 5, 5000
    plan = index_plan(spec)
    assert (plan.m, plan.ef_construction, plan.ef_search) == (100, 200, 1000)


def test_metric_from_1x_config_and_default():
    assert (
        _metric_of({"metadata": None, "configuration_json": {"hnsw": {"space": "cosine"}}})
        == Metric.COSINE
    )
    assert (
        _metric_of({"metadata": None, "configuration_json": {"spann": {"space": "ip"}}})
        == Metric.IP
    )
    assert _metric_of({"metadata": {}}) == Metric.L2


# -- snapshot + verify (no DB needed: snapshot is a Source, an in-memory fake is the Sink) -------


class MemorySink:
    uri = "memory://"

    def __init__(self, source: SnapshotSource, spec: CollectionSpec, metric: Metric | None = None):
        self.recs = {r.id: r for r in source._all(spec)}
        self.metric = metric or spec.metric  # a different one simulates a bad mapping

    def count(self, spec):
        return len(self.recs)

    def fetch(self, spec, ids):
        return [self.recs[i] for i in ids if i in self.recs]

    def query(self, spec, vector, k, *, exact=False):
        recs = list(self.recs.values())
        m = np.asarray([r.vector for r in recs], np.float32)
        d = distances(self.metric, np.asarray(vector, np.float32), m)
        return [recs[i].id for i in np.argsort(d, kind="stable")[:k]]


def _random_snapshot(tmp_path, n=300, dim=8, metric=Metric.COSINE):
    rng = np.random.default_rng(0)
    # Varied norms so cosine, l2 and ip rank neighbours differently.
    vecs = rng.normal(size=(n, dim)) * rng.uniform(0.1, 5.0, size=(n, 1))
    recs = [Record(f"id-{i}", vecs[i].tolist(), f"doc {i}", {"i": i}) for i in range(n)]

    class ListSource(SnapshotSource):
        def __init__(self):
            self.uri = "list://"

        def list_collections(self):
            return [CollectionSpec("c", dim, metric, n)]

        def scan(self, spec, cursor, batch_size):
            from vectorferry.core.model import Batch

            o = int(cursor or 0)
            page = recs[o : o + batch_size]
            return Batch(page, str(o + len(page)) if o + len(page) < n else None)

    src = ListSource()
    path = export_snapshot(src, tmp_path / "snap", src.list_collections(), batch_size=64)
    return SnapshotSource(path), recs


def test_snapshot_roundtrip_and_verify_passes(tmp_path):
    snap, recs = _random_snapshot(tmp_path)
    spec = snap.list_collections()[0]
    assert spec.count == len(recs)
    scanned, cursor = [], None
    while True:
        b = snap.scan(spec, cursor, 50)
        scanned += b.records
        cursor = b.next_cursor
        if cursor is None:
            break
    assert [record_hash(r) for r in scanned] == [record_hash(r) for r in recs]

    rep = verify_collection(snap, MemorySink(snap, spec), spec, sample=20, k=5, seed=1)
    assert rep.ok and rep.checked == len(recs)
    assert rep.metric_violations == 0 and rep.index_recall == 1.0 and rep.source_agreement == 1.0


@pytest.mark.parametrize("source_metric", list(Metric))
@pytest.mark.parametrize("sink_metric", list(Metric))
def test_verify_catches_wrong_similarity_measure(tmp_path, source_metric, sink_metric):
    snap, _ = _random_snapshot(tmp_path, metric=source_metric)
    spec = snap.list_collections()[0]
    rep = verify_collection(
        snap, MemorySink(snap, spec, sink_metric), spec, sample=30, k=10, seed=1
    )
    assert rep.checksums_ok  # the data itself is fine either way
    if source_metric == sink_metric:
        assert rep.metric_ok and rep.ok
    else:
        assert not rep.metric_ok and not rep.ok


def test_verify_catches_missing_and_changed(tmp_path):
    snap, _ = _random_snapshot(tmp_path)
    spec = snap.list_collections()[0]
    sink = MemorySink(snap, spec)
    del sink.recs["id-3"]
    changed = sink.recs["id-7"]
    sink.recs["id-7"] = Record(changed.id, changed.vector, "edited", changed.metadata)
    rep = verify_collection(snap, sink, spec, sample=5, k=5, seed=1)
    assert not rep.ok
    assert rep.missing == ["id-3"] and rep.mismatched == ["id-7"]
    assert not rep.counts_ok


def test_snapshot_detects_corrupt_file(tmp_path):
    snap, _ = _random_snapshot(tmp_path)
    part = next((snap.root / "c").glob("*.parquet"))
    part.write_bytes(part.read_bytes()[:-10] + b"0123456789")
    with pytest.raises(ValueError, match="corrupt"):
        SnapshotSource(snap.root)
