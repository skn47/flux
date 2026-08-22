"""
R5 walk-forward validation: persisted per-fold train/val/test date-range
boundaries (see `/home/h/.claude/plans/yes-kick-off-phase-stateful-trinket.md`,
"R5" section, and `backtest/walk_forward.py`, the orchestration script that
computes and writes these bounds once per fold before training that fold's
models).

New `walk_forward_bounds` table, in the SAME `data/events.db` every other
phase in this project uses, keyed `(fold_id, ticker, split)` ->
`(start_date, end_date)` -- the fold-scoped analogue of
`features/storage.py`'s existing `feature_split_bounds` table (one row per
`(ticker, split)`, no `fold_id`, for the single "headline" R1-R4 window).
Exists so `lstm.dataset.verify_no_leakage(fold_id=...)` has a persisted,
re-queryable source of truth per fold, not an in-memory assumption -- this
project's standing "verify against the DB, not in-memory assumptions"
discipline (see `features/README.md`'s "Verification" section, applied here
identically).

Per `(fold_id, ticker)`, all three rows (train/val/test) currently share the
SAME date ranges (all 8 tracked tickers share one trading calendar --
confirmed directly against the live DB, see `backtest/walk_forward.py`'s
boundary computation), but this table still stores one row per ticker rather
than a single fold-level row: (a) it matches `feature_split_bounds`'
existing per-ticker granularity exactly (an intentional consistency choice,
not an oversight), and (b) it would correctly tolerate a future ticker with
a different calendar without a schema change.

This module does NOT modify `features_daily` or `feature_split_bounds` --
per the task's explicit constraint, the single "headline" window's stored
split stays exactly as R1-R4 left it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "events.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS walk_forward_bounds (
    fold_id      INTEGER NOT NULL,
    ticker       TEXT NOT NULL,
    split        TEXT NOT NULL,
    start_date   TEXT NOT NULL,
    end_date     TEXT NOT NULL,
    n_label_days INTEGER NOT NULL,
    computed_at  TEXT NOT NULL,
    PRIMARY KEY (fold_id, ticker, split)
);
CREATE INDEX IF NOT EXISTS idx_walk_forward_bounds_fold ON walk_forward_bounds(fold_id);
"""


@dataclass
class WalkForwardBoundRow:
    fold_id: int
    ticker: str
    split: str              # "train" | "val" | "test"
    start_date: str
    end_date: str
    n_label_days: int        # usable-label-day count in this (fold, ticker, split)
    computed_at: str = ""


def ensure_schema(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()


def write_bounds(rows: Iterable[WalkForwardBoundRow], db_path: Path | str = DEFAULT_DB_PATH) -> int:
    """Idempotent upsert, same INSERT OR REPLACE convention as
    features/storage.py -- re-running fold-boundary computation overwrites
    stale rows rather than accumulating duplicates."""
    ensure_schema(db_path)
    rows = list(rows)
    payload = [
        (r.fold_id, r.ticker, r.split, r.start_date, r.end_date, r.n_label_days, r.computed_at)
        for r in rows
    ]
    if not payload:
        return 0
    conn = sqlite3.connect(str(db_path))
    conn.executemany(
        """
        INSERT OR REPLACE INTO walk_forward_bounds
            (fold_id, ticker, split, start_date, end_date, n_label_days, computed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    conn.commit()
    conn.close()
    return len(payload)


def write_fold_bounds_for_tickers(
    fold_id: int,
    tickers: list[str],
    split_boundaries: dict,
    n_label_days: dict,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    """
    Convenience wrapper: writes the SAME split_boundaries dict
    (`{"train"/"val"/"test": (start, end)}`, see lstm/dataset.py's R5
    docstring) as one row per (ticker, split) -- see this module's docstring
    for why per-ticker rows are stored even though the date ranges are
    currently identical across tickers. `n_label_days` is
    `{"train": n, "val": n, "test": n}`, the usable-label-day COUNT in each
    split (independent of the date-range span, since a date range can
    contain non-trading days) -- used for the hand-spot-check in
    `backtest/walk_forward.py`.
    """
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        WalkForwardBoundRow(
            fold_id=fold_id,
            ticker=ticker,
            split=split_name,
            start_date=bounds[0],
            end_date=bounds[1],
            n_label_days=n_label_days[split_name],
            computed_at=now,
        )
        for ticker in tickers
        for split_name, bounds in split_boundaries.items()
    ]
    return write_bounds(rows, db_path=db_path)


def read_bounds_frame(fold_id: int, db_path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Returns [ticker, split, start_date, end_date] for one fold_id -- same
    column shape lstm.dataset.verify_no_leakage's fold_id=None path reads
    from feature_split_bounds, so the same downstream assertion logic works
    unmodified regardless of which table it came from."""
    ensure_schema(db_path)
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        "SELECT ticker, split, start_date, end_date FROM walk_forward_bounds WHERE fold_id = ?",
        conn,
        params=(fold_id,),
    )
    conn.close()
    if df.empty:
        raise ValueError(f"walk_forward_bounds: no rows found for fold_id={fold_id}")
    return df


def verify_fold_boundaries(folds: list[dict]) -> dict:
    """
    Independent structural check (this project's "assert, don't just
    document" discipline, matching lstm/dataset.py::verify_no_leakage's
    pattern): given a list of fold boundary dicts (each keyed
    "train"/"val"/"test" -> (start_date, end_date), in fold order 1..K),
    asserts:
      1. WITHIN each fold: train_start <= train_end < val_start <= val_end
         < test_start <= test_end -- the three splits are in strictly
         increasing, non-overlapping chronological order.
      2. ACROSS folds: fold k's test_start is strictly AFTER fold (k-1)'s
         test_end -- test slices never overlap and never go backwards.
    Raises AssertionError on the first violation. Returns a small summary
    dict on success. Exists to catch an off-by-one in fold generation BEFORE
    any training/backtesting runs on faulty boundaries.
    """
    for i, f in enumerate(folds, start=1):
        tr_lo, tr_hi = f["train"]
        va_lo, va_hi = f["val"]
        te_lo, te_hi = f["test"]
        assert tr_lo <= tr_hi, f"fold {i}: train_start {tr_lo} > train_end {tr_hi}"
        assert tr_hi < va_lo, f"fold {i}: train_end {tr_hi} >= val_start {va_lo}"
        assert va_lo <= va_hi, f"fold {i}: val_start {va_lo} > val_end {va_hi}"
        assert va_hi < te_lo, f"fold {i}: val_end {va_hi} >= test_start {te_lo}"
        assert te_lo <= te_hi, f"fold {i}: test_start {te_lo} > test_end {te_hi}"

    for i in range(1, len(folds)):
        prev_test_end = folds[i - 1]["test"][1]
        this_test_start = folds[i]["test"][0]
        assert this_test_start > prev_test_end, (
            f"fold {i + 1}: test_start {this_test_start} does not come strictly after "
            f"fold {i}'s test_end {prev_test_end} -- overlapping or out-of-order test folds"
        )

    return {"n_folds_checked": len(folds), "status": "no fold-boundary violations found"}
