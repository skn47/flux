"""
SQLite-backed event store for Phase A ingestion.

Single table `raw_events`, one row per deduplicated RawEvent. Idempotency is
enforced at the database level via `INSERT OR IGNORE` keyed on the primary
key `id` (see ingestion/schema.py for how that id is derived) -- re-running
ingestion against the same sources will not create duplicate rows.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from src.ingestion.schema import RawEvent

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "events.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_events (
    id            TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    source_id     TEXT,
    published_at  TEXT NOT NULL,
    ingested_at   TEXT NOT NULL,
    title         TEXT,
    text          TEXT,
    url           TEXT,
    raw_metadata  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_raw_events_source ON raw_events(source);
CREATE INDEX IF NOT EXISTS idx_raw_events_published_at ON raw_events(published_at);
"""


class EventStore:
    """Thin wrapper around a sqlite3 connection to data/events.db."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def insert_events(self, events: Iterable[RawEvent]) -> tuple[int, int]:
        """
        Insert events, ignoring any whose `id` already exists.
        Returns (attempted, newly_inserted).
        """
        rows = [
            (
                e.id,
                e.source,
                e.source_id,
                e.published_at,
                e.ingested_at,
                e.title,
                e.text,
                e.url,
                json.dumps(e.raw_metadata, ensure_ascii=False, default=str),
            )
            for e in events
        ]
        if not rows:
            return (0, 0)

        before = self.count_rows()
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO raw_events
                (id, source, source_id, published_at, ingested_at, title, text, url, raw_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()
        after = self.count_rows()
        return (len(rows), after - before)

    def count_rows(self, source: str | None = None) -> int:
        if source:
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM raw_events WHERE source = ?", (source,)
            )
        else:
            cur = self.conn.execute("SELECT COUNT(*) FROM raw_events")
        return cur.fetchone()[0]

    def recent(self, source: str | None = None, limit: int = 10) -> list[sqlite3.Row]:
        """Fetch the most recently ingested rows, for sanity-checking."""
        if source:
            cur = self.conn.execute(
                """
                SELECT * FROM raw_events
                WHERE source = ?
                ORDER BY ingested_at DESC
                LIMIT ?
                """,
                (source, limit),
            )
        else:
            cur = self.conn.execute(
                "SELECT * FROM raw_events ORDER BY ingested_at DESC LIMIT ?",
                (limit,),
            )
        return cur.fetchall()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "EventStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
