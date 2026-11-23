# vectorferry

Move your vectors from ChromaDB to Postgres + pgvector, and **prove they arrived intact**.

Stuck on an old Chroma version because upgrading means a risky backup and restore? Want
your embeddings in the Postgres you already run? `vf` copies every collection, can resume
after a crash, and then checks the result: counts, a checksum of every record, the
similarity measure, and search quality.

```
$ vf copy --from chroma://localhost:8000 --to postgresql://me:pw@localhost/app

docs_cosine ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60000/60000
copied 1 collection(s) into postgresql://me:***@localhost/app

collection   counts          checksums  similarity measure  index recall@10  source agreement  result
docs_cosine  60000 = 60000   60000 ok   ok (100 queries)    0.996            0.974             OK
```

## Works with

| Source | Versions |
|---|---|
| ChromaDB server | 0.4.x, 0.5.x, 0.6.x, 1.x (talks HTTP directly, so the client version never has to match the server) |
| vectorferry snapshot | folders written by `vf export` |

| Target | |
|---|---|
| Postgres + pgvector | any pgvector with HNSW (0.5+) |

The phase 1 scope is Chroma → pgvector. pgvector → Chroma, Chroma → Chroma upgrades,
Qdrant and others come next. Each one is a single connector file.

## Install

```
pip install vectorferry        # or: uv tool install vectorferry
```

## Use

```bash
# See what's there and how each collection will land
vf inspect --from chroma://localhost:8000

# Copy (verifies at the end). Re-run the same command after a crash and it resumes.
vf copy --from chroma://localhost:8000 --to postgresql://user:pass@host:5432/db

# Just verify, e.g. in CI. Exits with 1 on any failure; writes a JSON report.
vf verify --from chroma://localhost:8000 --to postgresql://... --report verify.json

# Backup to portable Parquet, restore later (anywhere)
vf export --from chroma://localhost:8000 --out ./backup
vf import --from ./backup --to postgresql://...
```

Useful options: `-c/--collection` (repeatable), `--table-prefix` (default `vf_`),
`--table-map docs=my_docs`, `--pg-schema`, `--batch-size`, `--no-index`, `--restart`.
For Chroma 0.6+ use `--tenant` and `--database`; for auth, set the `VF_CHROMA_TOKEN`
environment variable.

## What lands in Postgres

One table per collection:

```sql
id text PRIMARY KEY, embedding vector(<dim>), document text, metadata jsonb, _vf_hash bigint
```

- **Similarity measure:** Chroma's `hnsw:space` picks the index and operator:
  - `cosine` → `vector_cosine_ops` / `<=>`
  - `l2` → `vector_l2_ops` / `<->`
  - `ip` → `vector_ip_ops` / `<#>`
- **Index settings** (M, ef_construction) are copied from Chroma, so search quality
  carries over. The table comment says which `SET hnsw.ef_search = N` gives your app the
  same quality.
- **More than 2000 dimensions:** indexed as `halfvec`. Over 4000: no index (exact
  search), with a warning.
- **Metadata** becomes `jsonb`, so `WHERE metadata->>'source' = 'faq'` just works.

## How verification works

| Check | Pass rule |
|---|---|
| Counts | equal |
| Checksums | every record's id, vector (float32), document and metadata re-hashed on **both** sides. Nothing is trusted from what was written. |
| Similarity measure | for sample queries, pgvector's *exact* top-k is re-scored with Chroma's measure. It must come back in order, and nothing Chroma returns may be closer. A cosine↔l2 mix-up fails immediately. |
| Index recall | pgvector's HNSW against its own exact search: ≥ `--min-recall` (0.95), or no worse than Chroma's own recall |

Why not just compare Chroma's results with pgvector's? Both are approximate, so the
comparison mixes their errors together. In testing, Chroma 0.4's defaults found only 81%
of the true l2 neighbours. The "source agreement" column shows that number for
information.

## Develop

```bash
uv sync
uv run pytest                                  # unit tests
docker compose -f tests/compose.yml up -d      # Chroma 0.4 / 0.6 / 1.0 + pgvector
uv run pytest -m integration
```

## License

Apache-2.0
