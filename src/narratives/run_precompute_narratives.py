"""
One-time (re-runnable) batch job: generates a short AI narrative for every
event that has ever ranked in a ticker's daily top-5 attribution -- the
bounded, worth-explaining set the 3D causality-graph feature needs (see
labeling/README.md and the plan this shipped under).

Candidate set is exactly `SELECT DISTINCT event_id FROM flux_attribution_daily
WHERE event_id NOT IN (SELECT event_id FROM event_narratives)` joined to
event_catalog for title/event_type -- flux_attribution_daily already only
ever stores the top-K=5 rows per (ticker, date), so this IS the 774-event
set, no extra filtering needed. Idempotent by construction: unlike
labeling/run_gdelt_reclassify.py's multi-verdict state machine (which needs a
separate gdelt_content_checked table), a narrative has exactly one terminal
state -- generated or not -- so event_narratives itself is the marker.

Safe to re-run any time (e.g. after a new event gets promoted into a top-5
slot); only ever processes rows not yet narrated.

Disclosed gap, found and fixed 2026-08-19: this module was never wired into
scripts/refresh_pipeline.sh, so every flux_engine.run_timeseries rebuild
could promote new events into a ticker's top-5 attribution with nothing to
narrate them -- same shape as labeling/README.md bug 11 (run_gdelt_reclassify
had the identical gap for the GDELT recheck), just not caught for this table
at the time it shipped. Surfaced once a daily cron started running the
pipeline (see api/README.md's "Data freshness" section) instead of rarely by
hand: the causality graph started showing "AI explanation not yet generated"
for events that had one before. Fixed by adding this as a pipeline step,
right after the second flux_engine.run_timeseries rerun.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3

from src.ingestion.storage import DEFAULT_DB_PATH
from src.narratives.ollama_narrator import generate_batch
from src.narratives.storage import NarrativeStore

logger = logging.getLogger("narratives.run_precompute_narratives")


def _candidate_rows(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT DISTINCT a.event_id, c.title, c.event_type,
               (SELECT corridor FROM flux_attribution_daily a2
                WHERE a2.event_id = a.event_id LIMIT 1) AS corridor
        FROM flux_attribution_daily a
        JOIN event_catalog c ON c.event_id = a.event_id
        WHERE a.event_id NOT IN (SELECT event_id FROM event_narratives)
        """
    ).fetchall()
    out = []
    for event_id, title, event_type, corridor_json in rows:
        try:
            corridor = json.loads(corridor_json) if corridor_json else []
        except ValueError:
            corridor = []
        out.append({"event_id": event_id, "title": title, "event_type": event_type, "corridor": corridor})
    return out


def run(db_path=DEFAULT_DB_PATH, limit: int | None = None) -> dict:
    conn = sqlite3.connect(db_path)
    store = NarrativeStore(db_path)
    try:
        candidates = _candidate_rows(conn)
        total_candidates = len(candidates)
        logger.info("run_precompute_narratives: %d unnarrated candidates", total_candidates)

        batch = candidates[:limit] if limit else candidates
        results = generate_batch(batch)
        for n in results:
            store.insert_narrative(n)

        return {
            "total_candidates": total_candidates,
            "attempted": len(batch),
            "narrated": len(results),
            "failed": len(batch) - len(results),
            "total_narrated_in_db": store.count_rows(),
        }
    finally:
        store.close()
        conn.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="cap batch size (for testing)")
    args = parser.parse_args()

    summary = run(limit=args.limit)

    print(f"Unnarrated candidates:   {summary['total_candidates']}")
    print(f"Attempted this run:      {summary['attempted']}")
    print(f"Narrated:                {summary['narrated']}")
    print(f"Failed/skipped:          {summary['failed']}")
    print(f"Total narrated in DB:    {summary['total_narrated_in_db']}")


if __name__ == "__main__":
    main()
