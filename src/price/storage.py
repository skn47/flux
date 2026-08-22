"""
SQLite-backed daily-bar store for the price ingestion package.

Lives in the SAME `data/events.db` file used by Phase A/B/D/E (raw_events,
classified_events, ...) rather than a separate `data/prices.db`. Rationale
(see price/README.md for the fuller writeup):
  - every other phase in this project already shares one DB file; splitting
    price data out would be the first exception, not a continuation of an
    existing pattern.
  - Phase F's whole point is joining flux_score(stock, t) against price
    history for the same (ticker, date) pairs -- keeping both in one file
    means that join is a single sqlite3 connection with no ATTACH DATABASE
    dance.
  - the tables don't collide: `daily_prices` is a new table, independent
    primary key, no foreign keys into raw_events/classified_events required
    at this stage.

New table `daily_prices`, one row per (ticker, date). Idempotency is enforced
at the database level via `INSERT OR IGNORE` keyed on the composite primary
key `(ticker, date)` -- re-running ingestion against the same tickers/date
range will not create duplicate rows and will not overwrite existing ones.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from src.price.schema import PriceBar

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "events.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_prices (
    ticker        TEXT NOT NULL,
    date          TEXT NOT NULL,
    open          REAL NOT NULL,
    high          REAL NOT NULL,
    low           REAL NOT NULL,
    close         REAL NOT NULL,
    adj_close     REAL NOT NULL,
    volume        INTEGER NOT NULL,
    source        TEXT NOT NULL DEFAULT 'yfinance',
    ingested_at   TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date);
"""


class PriceStore:
    """Thin wrapper around a sqlite3 connection to data/events.db."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def insert_bars(self, bars: Iterable[PriceBar]) -> tuple[int, int]:
        """
        Insert bars, ignoring any whose (ticker, date) already exists.
        Returns (attempted, newly_inserted).
        """
        rows = [
            (
                b.ticker,
                b.date,
                b.open,
                b.high,
                b.low,
                b.close,
                b.adj_close,
                b.volume,
                b.source,
                b.ingested_at,
            )
            for b in bars
        ]
        if not rows:
            return (0, 0)

        before = self.count_rows()
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO daily_prices
                (ticker, date, open, high, low, close, adj_close, volume, source, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()
        after = self.count_rows()
        return (len(rows), after - before)

    def count_rows(self, ticker: str | None = None) -> int:
        if ticker:
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM daily_prices WHERE ticker = ?", (ticker,)
            )
        else:
            cur = self.conn.execute("SELECT COUNT(*) FROM daily_prices")
        return cur.fetchone()[0]

    def date_range(self, ticker: str) -> tuple[Optional[str], Optional[str]]:
        """(min_date, max_date) for a ticker, or (None, None) if it has no rows."""
        cur = self.conn.execute(
            "SELECT MIN(date), MAX(date) FROM daily_prices WHERE ticker = ?",
            (ticker,),
        )
        row = cur.fetchone()
        return (row[0], row[1])

    def recent(self, ticker: str | None = None, limit: int = 10) -> list[sqlite3.Row]:
        """Fetch the most recent rows by date, for sanity-checking."""
        if ticker:
            cur = self.conn.execute(
                "SELECT * FROM daily_prices WHERE ticker = ? ORDER BY date DESC LIMIT ?",
                (ticker, limit),
            )
        else:
            cur = self.conn.execute(
                "SELECT * FROM daily_prices ORDER BY date DESC LIMIT ?", (limit,)
            )
        return cur.fetchall()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "PriceStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
