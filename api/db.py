"""
Read-only SQLite access for the API. Every connection is opened in SQLite's
own `mode=ro` URI mode -- this makes it structurally impossible for a bug in
this process to write to data/events.db, enforced by SQLite itself, not by
code discipline alone.

A fresh, short-lived connection is opened per call rather than one held
connection, so a request never holds a lock across the response cycle (cheap
for SQLite; this API's endpoints are all small point/range lookups against
small indexed tables -- see api/README.md for which tables are safe to touch
here and why raw_events/classified_events are never queried by this package).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from api.config import DB_PATH

# data/events.db was migrated to WAL journal mode (Phase H, H4) specifically
# so a manual pipeline rerun (a writer) doesn't block concurrent API reads --
# WAL readers see a consistent snapshot without waiting on an in-progress
# writer. busy_timeout is still set as a defense-in-depth fallback (e.g. the
# brief moment a WAL checkpoint itself takes), not the primary mitigation.
_BUSY_TIMEOUT_MS = 5000


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=_BUSY_TIMEOUT_MS / 1000)
    con.row_factory = sqlite3.Row
    con.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    try:
        yield con
    finally:
        con.close()


def query(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    with get_connection() as con:
        return con.execute(sql, params).fetchall()


def query_one(sql: str, params: tuple = ()) -> sqlite3.Row | None:
    with get_connection() as con:
        return con.execute(sql, params).fetchone()
