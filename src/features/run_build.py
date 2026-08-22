"""
CLI entry point: build the model-ready feature table for every tracked
ticker and store it in `data/events.db` (`features_daily` +
`feature_split_bounds`). Mirrors `price/run_ingest.py`'s per-ticker error
isolation and printed summary style.

Usage:
    ./.venv/bin/python -m features.run_build
"""

from __future__ import annotations

import sqlite3
import sys

from config.mvp_scope import TRACKED_TICKERS
from src.features.build import build_all
from src.features.storage import DEFAULT_DB_PATH, FeatureStore


def main() -> int:
    conn = sqlite3.connect(str(DEFAULT_DB_PATH))
    store = FeatureStore(DEFAULT_DB_PATH)

    results, errors = build_all(conn, TRACKED_TICKERS)

    print(f"{'ticker':<8}{'rows':>7}{'min_date':>12}{'max_date':>12}"
          f"{'train':>8}{'val':>8}{'test':>8}{'ffill':>8}{'unmatched':>10}{'warmup_nan':>11}")
    total_rows = 0
    total_inserted = 0
    for ticker in TRACKED_TICKERS:
        if ticker in errors:
            print(f"{ticker:<8} ERROR: {errors[ticker]}")
            continue
        rows, bounds, diag = results[ticker]
        attempted, inserted = store.insert_features(rows)
        store.insert_split_bounds(bounds)
        total_rows += len(rows)
        total_inserted += inserted

        b = {sb.split: sb for sb in bounds}
        min_d, max_d = rows[0].date, rows[-1].date
        print(
            f"{ticker:<8}{len(rows):>7}{min_d:>12}{max_d:>12}"
            f"{b['train'].n_rows:>8}{b['val'].n_rows:>8}{b['test'].n_rows:>8}"
            f"{diag.n_forward_filled_flux_matches:>8}{diag.n_unmatched_dropped:>10}"
            f"{diag.warmup_nan_rows_dropped:>11}"
        )

    if errors:
        print(f"\n{len(errors)}/{len(TRACKED_TICKERS)} ticker(s) failed:")
        for ticker, exc in errors.items():
            print(f"  {ticker}: {exc!r}")

    print(f"\nTotal feature rows built: {total_rows}")
    print(f"Total newly inserted/updated rows in features_daily: {total_inserted}")
    print(f"DB path: {DEFAULT_DB_PATH}")

    store.close()
    conn.close()

    if len(errors) == len(TRACKED_TICKERS):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
