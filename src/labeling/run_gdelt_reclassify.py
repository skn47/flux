"""
Orchestrates the two-stage content re-check of `gdelt_derived` events
currently labeled `geopolitical_military_tension`, per
progress/event_taxonomy.md's flagged QuadClass-blindness discovery.

Stage 1 (labeling/gdelt_content_filter.py, free, instant): confident title
keyword matches resolve without touching the network.
Stage 2 (labeling/ollama_labeler.py, free, local, slow): everything Stage 1
can't confidently call goes to a local LLM for a real content read.

Correction semantics (redesigned 2026-08-16 -- see labeling/README.md):
  - Stage 1 "reject": DELETE the original `gdelt_derived` row (it stops
    competing for its old, inflated cluster bucket). This is the only
    no-Ollama-cost fast path left -- it's been repeatedly quantified and
    refined (see labeling/README.md bugs 3, 7, 9, 11) and stays a legitimate
    cheap filter.
  - Stage 1 "confirm" NO LONGER short-circuits. It used to be trusted
    outright with zero numeric score -- the direct root cause of at least two
    false-positive bugs this session. It now routes into the same Stage 2
    batch as "uncertain" rows, so every candidate that isn't Stage-1-rejected
    gets a real, content-based confidence score before it can count.
  - Stage 2 landing on "unclassified", OR "geopolitical_military_tension"
    with confidence below `ollama_labeler.MIN_CONFIRM_CONFIDENCE`: DELETE the
    original `gdelt_derived` row, same as a Stage 1 reject. Stage 2's
    event_type is a binary {geopolitical_military_tension, unclassified} --
    see labeling/ollama_labeler.py's module docstring for why it no longer
    attempts to guess a specific replacement category: two earlier attempts
    let the local model pick among the full taxonomy and both were unreliable
    (e.g. car crashes and shootings landing on "natural_disaster"). So a
    Stage 2 correction is always a plain delete, never a re-typed insert.
  - Stage 2 confirming geopolitical_military_tension AT OR ABOVE the
    confidence threshold: the original `gdelt_derived` row is left in place
    AND an `ollama_assisted` row is inserted alongside it -- both now compete
    fairly in flux_engine.timeseries.build_clusters()'s max-K selection, so
    the better-calibrated (content-aware) confidence naturally wins out over
    gdelt_labeler.py's blind fixed 0.55 if it's higher, with no special-casing
    needed here. Critically, `flux_engine/query.py::load_events()` now also
    requires a `confirm_stage2` verdict for any `gdelt_derived`
    `geopolitical_military_tension` row to count at all (a structural,
    default-deny gate) -- so a row that's merely "left in place" here but
    never reaches `confirm_stage2` is invisible regardless of clustering.

Idempotent re-run: every event_id this script looks at gets recorded in a new
`gdelt_content_checked` table (verdict + timestamp) so a second run only
picks up rows that weren't checked yet -- e.g. round 2's newly-promoted
cluster representatives after flux_engine.run_timeseries re-clusters, or rows
that failed to process last time because Ollama was down (see below).

If Ollama is unreachable, Stage 2 rows are deliberately left UNCHECKED (not
marked in gdelt_content_checked) so a future run retries them -- unlike
Stage 1, whose confirm/reject verdicts don't depend on anything that can be
"temporarily down."
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.ingestion.storage import DEFAULT_DB_PATH
from src.labeling.gdelt_content_filter import classify_title
from src.labeling.ollama_labeler import MIN_CONFIRM_CONFIDENCE, label_batch
from src.labeling.storage import LabelStore

logger = logging.getLogger("labeling.run_gdelt_reclassify")

_CHECKED_SCHEMA = """
CREATE TABLE IF NOT EXISTS gdelt_content_checked (
    event_id    TEXT PRIMARY KEY,
    verdict     TEXT NOT NULL,   -- confirm_stage1 | reject_stage1 | confirm_stage2 | reject_stage2
    checked_at  TEXT NOT NULL
);
"""


@dataclass
class ReclassifySummary:
    total_candidates: int = 0
    stage1_confirm: int = 0  # diagnostic only since the 2026-08-16 redesign --
                              # these rows are still routed to Stage 2, not a final verdict
    stage1_reject: int = 0
    stage1_uncertain: int = 0
    stage2_available: bool = False
    stage2_attempted: int = 0
    stage2_confirm: int = 0
    stage2_reject: int = 0
    stage2_reject_low_confidence: int = 0  # subset of stage2_reject: model said
                                            # geopolitical_military_tension but confidence < MIN_CONFIRM_CONFIDENCE
    stage2_failed: int = 0
    examples: list = field(default_factory=list)  # (event_id, before_type, after_type, title)


def _candidate_rows(conn: sqlite3.Connection) -> list[dict]:
    conn.executescript(_CHECKED_SCHEMA)
    rows = conn.execute(
        """
        SELECT c.event_id, c.title
        FROM event_catalog c
        WHERE c.label_source = 'gdelt_derived'
          AND c.event_type = 'geopolitical_military_tension'
          AND c.title IS NOT NULL
          AND c.event_id NOT IN (SELECT event_id FROM gdelt_content_checked)
        """
    ).fetchall()
    return [{"id": r[0], "title": r[1]} for r in rows]


def _mark_checked(conn: sqlite3.Connection, event_id: str, verdict: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO gdelt_content_checked (event_id, verdict, checked_at) VALUES (?, ?, ?)",
        (event_id, verdict, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def run(db_path=DEFAULT_DB_PATH, max_stage2: int | None = None) -> ReclassifySummary:
    summary = ReclassifySummary()
    conn = sqlite3.connect(db_path)
    store = LabelStore(db_path)
    try:
        candidates = _candidate_rows(conn)
        summary.total_candidates = len(candidates)
        logger.info("run_gdelt_reclassify: %d unchecked candidates", len(candidates))

        to_stage2: list[dict] = []
        for row in candidates:
            verdict = classify_title(row["title"])
            if verdict == "reject":
                summary.stage1_reject += 1
                store.delete_label(row["id"], "gdelt_derived")
                _mark_checked(conn, row["id"], "reject_stage1")
                if len(summary.examples) < 20:
                    summary.examples.append((row["id"], "geopolitical_military_tension", "unclassified (stage1)", row["title"]))
            else:
                # "confirm" no longer short-circuits (see module docstring's
                # 2026-08-16 redesign note) -- both "confirm" and "uncertain"
                # go to Stage 2 for a real, scored decision.
                if verdict == "confirm":
                    summary.stage1_confirm += 1
                else:
                    summary.stage1_uncertain += 1
                to_stage2.append(row)

        if to_stage2:
            batch = to_stage2[:max_stage2] if max_stage2 else to_stage2
            logger.info(
                "run_gdelt_reclassify: sending %d/%d stage-1-confirmed/uncertain rows to stage 2",
                len(batch), len(to_stage2),
            )
            summary.stage2_attempted = len(batch)
            results = label_batch(batch)
            summary.stage2_available = bool(results) or len(batch) == 0
            labeled_ids = {r.event_id for r in results}

            for r in results:
                if r.event_type == "geopolitical_military_tension" and r.confidence >= MIN_CONFIRM_CONFIDENCE:
                    summary.stage2_confirm += 1
                    store.insert_labels([r])
                    _mark_checked(conn, r.event_id, "confirm_stage2")
                else:
                    summary.stage2_reject += 1
                    if r.event_type == "geopolitical_military_tension":
                        summary.stage2_reject_low_confidence += 1
                    store.delete_label(r.event_id, "gdelt_derived")
                    _mark_checked(conn, r.event_id, "reject_stage2")
                    if len(summary.examples) < 20:
                        title = next((row["title"] for row in batch if row["id"] == r.event_id), "")
                        summary.examples.append((r.event_id, "geopolitical_military_tension", r.event_type, title))

            summary.stage2_failed = len(batch) - len(labeled_ids)
            if not results and batch:
                logger.warning(
                    "run_gdelt_reclassify: stage 2 returned no results for %d rows "
                    "(Ollama unavailable?) -- leaving them unchecked for retry", len(batch)
                )
    finally:
        store.close()
        conn.close()

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-stage2", type=int, default=None, help="cap stage-2 batch size (for testing)")
    args = parser.parse_args()

    summary = run(max_stage2=args.max_stage2)

    print(f"Candidates considered:        {summary.total_candidates}")
    print(f"Stage 1 confirm (-> stage 2): {summary.stage1_confirm}")
    print(f"Stage 1 reject:               {summary.stage1_reject}")
    print(f"Stage 1 uncertain (-> stage 2):{summary.stage1_uncertain}")
    print(f"Stage 2 attempted:            {summary.stage2_attempted}")
    print(f"Stage 2 confirm (>= {MIN_CONFIRM_CONFIDENCE}):     {summary.stage2_confirm}")
    print(f"Stage 2 reject:               {summary.stage2_reject}")
    print(f"  (of which, low-confidence): {summary.stage2_reject_low_confidence}")
    print(f"Stage 2 failed/skipped:       {summary.stage2_failed}")
    print()
    print("Example corrections:")
    for event_id, before, after, title in summary.examples:
        print(f"  [{event_id[:12]}] {before} -> {after} :: {title}")


if __name__ == "__main__":
    main()
