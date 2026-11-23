"""Seed a Chroma server (any version) with synthetic collections over raw HTTP.

uv run python tests/seed.py http://localhost:18004 --n 5000 --dim 64
"""

from __future__ import annotations

import argparse

import httpx
import numpy as np


def base(http: httpx.Client) -> str:
    if http.get("/api/v2/heartbeat").status_code == 200:
        return "/api/v2/tenants/default_tenant/databases/default_database/collections"
    return "/api/v1/collections"


def seed(url: str, name: str, n: int, dim: int, metric: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    with httpx.Client(base_url=url, timeout=120) as http:
        b = base(http)
        for c in http.get(b).json():
            if c["name"] == name:
                http.delete(f"{b}/{name}")  # v1 deletes by name; v2 too
        r = http.post(b, json={"name": name, "metadata": {"hnsw:space": metric}})
        r.raise_for_status()
        cid = r.json()["id"]
        for start in range(0, n, 1000):
            ids = [f"{name}-{i}" for i in range(start, min(n, start + 1000))]
            vecs = rng.normal(size=(len(ids), dim)).astype(np.float32)
            docs = [None if i % 10 == 0 else f"document {i} — ünïcödé ✓" for i in range(len(ids))]
            metas = [
                None
                if i % 7 == 0
                else {
                    "i": start + i,
                    "even": (start + i) % 2 == 0,
                    "score": round(float(rng.random()), 4),
                    "tag": "t" * (i % 3),
                }
                for i in range(len(ids))
            ]
            r = http.post(
                f"{b}/{cid}/add",
                json={
                    "ids": ids,
                    "embeddings": vecs.tolist(),
                    "documents": docs,
                    "metadatas": metas,
                },
            )
            r.raise_for_status()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--dim", type=int, default=64)
    a = p.parse_args()
    for metric in ("cosine", "l2", "ip"):
        seed(a.url, f"docs_{metric}", a.n, a.dim, metric)
    print("seeded", a.url)
