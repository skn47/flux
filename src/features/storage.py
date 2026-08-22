"""
SQLite-backed store for the model-ready feature table, in the SAME
`data/events.db` file every other phase in this project uses (raw_events,
classified_events, daily_prices, flux_scores_daily, ...). See
features/README.md ("Storage") for the full rationale; short version: this
step's entire job is joining `daily_prices` against `flux_scores_daily`,
both already in `events.db` -- a separate file would need `ATTACH DATABASE`
for no benefit, and would be the first exception to an established pattern.

Two new tables:
  - `features_daily`: one row per (ticker, date) -- the joined, unscaled
    feature vector plus its split assignment. Idempotent write via
    `INSERT OR REPLACE` keyed on (ticker, date), matching
    `flux_engine/timeseries.py`'s convention (not `daily_prices`'
    INSERT-OR-IGNORE) because these are DERIVED/recomputable values -- if the
    indicator formulas or split ratios ever change, re-running this build
    should overwrite stale rows, not silently keep old ones around forever.
  - `feature_split_bounds`: one row per (ticker, split) recording the actual
    date-range boundaries and usable-label counts used, so the 3-way
    chronological split is reproducible from the DB alone without
    re-deriving it from row order.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from src.features.schema import FeatureRow, SplitBounds

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "events.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS features_daily (
    ticker                    TEXT NOT NULL,
    date                      TEXT NOT NULL,
    adj_close                 REAL NOT NULL,
    daily_return              REAL NOT NULL,
    sma_short                 REAL NOT NULL,
    sma_long                  REAL NOT NULL,
    volatility_10             REAL NOT NULL,
    flux_score               REAL NOT NULL,
    flux_score_source_date   TEXT NOT NULL,
    split                     TEXT NOT NULL,
    computed_at               TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_features_daily_date ON features_daily(date);
CREATE INDEX IF NOT EXISTS idx_features_daily_split ON features_daily(ticker, split);

CREATE TABLE IF NOT EXISTS feature_split_bounds (
    ticker                  TEXT NOT NULL,
    split                   TEXT NOT NULL,
    start_date              TEXT NOT NULL,
    end_date                TEXT NOT NULL,
    n_rows                  INTEGER NOT NULL,
    lookback_days           INTEGER NOT NULL,
    first_usable_label_date TEXT,
    n_usable_labels         INTEGER NOT NULL,
    computed_at             TEXT NOT NULL,
    PRIMARY KEY (ticker, split)
);
"""


class FeatureStore:
    """Thin wrapper around a sqlite3 connection to data/events.db."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def insert_features(self, rows: Iterable[FeatureRow]) -> tuple[int, int]:
        """Upsert feature rows. Returns (attempted, rows_after - rows_before)."""
        rows = list(rows)
        payload = [
            (
                r.ticker, r.date, r.adj_close, r.daily_return, r.sma_short,
                r.sma_long, r.volatility_10, r.flux_score,
                r.flux_score_source_date, r.split, r.computed_at,
            )
            for r in rows
        ]
        if not payload:
            return (0, 0)

        before = self.count_rows()
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO features_daily
                (ticker, date, adj_close, daily_return, sma_short, sma_long,
                 volatility_10, flux_score, flux_score_source_date, split, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        self.conn.commit()
        after = self.count_rows()
        return (len(payload), after - before)

    def insert_split_bounds(self, bounds: Iterable[SplitBounds]) -> int:
        payload = [
            (
                b.ticker, b.split, b.start_date, b.end_date, b.n_rows,
                b.lookback_days, b.first_usable_label_date, b.n_usable_labels,
                b.computed_at,
            )
            for b in bounds
        ]
        if not payload:
            return 0
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO feature_split_bounds
                (ticker, split, start_date, end_date, n_rows, lookback_days,
                 first_usable_label_date, n_usable_labels, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        self.conn.commit()
        return len(payload)

    def count_rows(self, ticker: str | None = None) -> int:
        if ticker:
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM features_daily WHERE ticker = ?", (ticker,)
            )
        else:
            cur = self.conn.execute("SELECT COUNT(*) FROM features_daily")
        return cur.fetchone()[0]

    def date_range(self, ticker: str) -> tuple[str | None, str | None]:
        cur = self.conn.execute(
            "SELECT MIN(date), MAX(date) FROM features_daily WHERE ticker = ?", (ticker,)
        )
        row = cur.fetchone()
        return (row[0], row[1])

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "FeatureStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
