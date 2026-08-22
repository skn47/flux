"""
Chunked GDELT labeling driver for the historical backfill.

Why this exists alongside labeling/run_label.py
------------------------------------------------
`labeling.run_label` (and LabelStore.unlabeled_raw_events) materializes ALL
not-yet-labeled rows for a source into memory at once. That is fine for the
live connector's handful of rows, but the GDELT 1.0 daily backfill inserts
millions of raw_events rows, and this box has only a few GB of free RAM --
loading every row as a Python dict would OOM.

This driver does exactly the same labeling (it calls the SAME
`labeling.gdelt_labeler.label_batch` and the SAME `LabelStore.insert_labels`,
so the resulting classified_events rows are byte-for-byte what run_label would
produce) but streams `source='gdelt'` rows in rowid-keyed batches, holding only
one batch in memory at a time. It is idempotent: INSERT OR IGNORE on
(event_id, label_source='gdelt_derived') means already-labeled rows are
skipped, and an anti-join per batch avoids recomputing labels for rows already
done, so re-running is cheap and safe.

Run with:
    ./.venv/bin/python -m labeling.label_gdelt_backfill
"""

from __future__ import annotations

import json
import logging
import sys
import time

from src.labeling import gdelt_labeler
from src.labeling.storage import LabelStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("labeling.label_gdelt_backfill")

LABEL_SOURCE = "gdelt_derived"
BATCH_SIZE = 20000


def _fetch_batch(store: LabelStore, last_rowid: int, batch_size: int) -> list[dict]:
    """rowid-keyed page of source='gdelt' rows NOT yet labeled gdelt_derived,
    with raw_metadata already json.loads'd. Keyset pagination (rowid > last)
    avoids OFFSET's O(n) scan cost across millions of rows."""
    cur = store.conn.execute(
        """
        SELECT r.rowid AS _rowid, r.id, r.raw_metadata
        FROM raw_events r
        LEFT JOIN classified_events c
            ON r.id = c.event_id AND c.label_source = ?
        WHERE r.source = 'gdelt' AND r.rowid > ? AND c.event_id IS NULL
        ORDER BY r.rowid
        LIMIT ?
        """,
        (LABEL_SOURCE, last_rowid, batch_size),
    )
    rows = []
    for row in cur.fetchall():
        d = {"_rowid": row["_rowid"], "id": row["id"]}
        try:
            d["raw_metadata"] = json.loads(row["raw_metadata"] or "{}")
        except (TypeError, ValueError):
            d["raw_metadata"] = {}
        rows.append(d)
    return rows


def _max_gdelt_rowid(store: LabelStore) -> int:
    cur = store.conn.execute(
        "SELECT COALESCE(MAX(rowid), 0) FROM raw_events WHERE source='gdelt'"
    )
    return cur.fetchone()[0]


def run(batch_size: int = BATCH_SIZE) -> tuple[int, int]:
    """Label all unlabeled gdelt rows in batches. Returns
    (candidates_seen, newly_inserted)."""
    store = LabelStore()
    t0 = time.time()
    before = store.count_rows(label_source=LABEL_SOURCE)
    max_rowid = _max_gdelt_rowid(store)

    total_candidates = 0
    total_inserted = 0
    last_rowid = 0
    try:
        while True:
            batch = _fetch_batch(store, last_rowid, batch_size)
            if not batch:
                break
            last_rowid = batch[-1]["_rowid"]
            labels = gdelt_labeler.label_batch(batch)
            _, inserted = store.insert_labels(labels)
            total_candidates += len(batch)
            total_inserted += inserted
            logger.info(
                "labeled batch up to rowid %d/%d: +%d candidates (+%d inserted), "
                "running totals %d/%d",
                last_rowid, max_rowid, len(batch), inserted,
                total_candidates, total_inserted,
            )
    finally:
        after = store.count_rows(label_source=LABEL_SOURCE)
        store.close()

    logger.info(
        "done: %d candidates processed, %d newly inserted, gdelt_derived rows %d -> %d, %.1fs",
        total_candidates, total_inserted, before, after, time.time() - t0,
    )
    return total_candidates, total_inserted


if __name__ == "__main__":
    run()
    sys.exit(0)
