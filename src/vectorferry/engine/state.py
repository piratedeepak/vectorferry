"""Resume state in a local SQLite file.

Per (job, collection) it keeps the source cursor and how many records were copied. The
cursor is saved only after the sink has committed the batch, so a crash can at worst
redo one batch, and sinks upsert, so redoing it is harmless.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Progress:
    cursor: str | None
    copied: int
    done: bool


class State:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS progress (
                 job TEXT NOT NULL, collection TEXT NOT NULL,
                 cursor TEXT, copied INTEGER NOT NULL DEFAULT 0, done INTEGER NOT NULL DEFAULT 0,
                 updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                 PRIMARY KEY (job, collection))"""
        )
        self._db.commit()

    def get(self, job: str, collection: str) -> Progress | None:
        row = self._db.execute(
            "SELECT cursor, copied, done FROM progress WHERE job=? AND collection=?",
            (job, collection),
        ).fetchone()
        return Progress(row[0], row[1], bool(row[2])) if row else None

    def save(self, job: str, collection: str, cursor: str | None, copied: int, done: bool) -> None:
        self._db.execute(
            """INSERT INTO progress (job, collection, cursor, copied, done) VALUES (?,?,?,?,?)
               ON CONFLICT (job, collection) DO UPDATE SET cursor=excluded.cursor,
                 copied=excluded.copied, done=excluded.done, updated_at=CURRENT_TIMESTAMP""",
            (job, collection, cursor, copied, int(done)),
        )
        self._db.commit()

    def reset(self, job: str, collection: str) -> None:
        self._db.execute("DELETE FROM progress WHERE job=? AND collection=?", (job, collection))
        self._db.commit()

    def close(self) -> None:
        self._db.close()
