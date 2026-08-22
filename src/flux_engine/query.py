"""
Loads real (raw_events JOIN classified_events) rows from data/events.db into the
plain dicts flux_engine.formula expects. Kept separate from formula.py so the
scoring math has zero DB/IO dependency and can be unit-tested/hand-verified in
isolation (see flux_engine/README.md for the worked-example verification).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.ingestion.storage import DEFAULT_DB_PATH


def _parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def load_events(
    db_path: Path | str = DEFAULT_DB_PATH,
    label_sources: Optional[list[str]] = None,
    since: Optional[datetime] = None,
) -> list[dict]:
    """
    Returns events shaped for flux_engine.formula.flux_score(): event_id,
    label_source, event_type, severity_score, confidence, polarity, countries
    (parsed from JSON), published_at (tz-aware datetime, from raw_events, per
    Phase A's published_at/ingested_at discipline -- never ingested_at here).

    `since` is an optional prefilter (e.g. now - lookback_window) purely for
    query efficiency; the formula's own time_decay/epsilon cutoff is the real
    correctness guard, this is just avoiding pulling stale rows into memory.
    """
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        # Default-deny gate for gdelt_derived geopolitical_military_tension
        # (added 2026-08-16, see labeling/README.md): gdelt_labeler.py assigns
        # that event_type from GDELT's QuadClass alone, with zero content
        # signal -- it must not count toward flux_score, clustering, or the
        # feed until labeling/run_gdelt_reclassify.py's Stage 2 content check
        # has actually confirmed it above ollama_labeler.MIN_CONFIRM_CONFIDENCE
        # (recorded as a 'confirm_stage2' verdict in gdelt_content_checked).
        # This is what makes "discard unless it passes" permanent rather than
        # dependent on someone re-running the reclassify script after every
        # flux_engine.run_timeseries rebuild -- a gap that caused two rounds
        # of recurring false positives earlier the same day this was added.
        # Every other event_type/label_source combination is unaffected.
        gate = (
            "AND (c.label_source != 'gdelt_derived' "
            "OR c.event_type != 'geopolitical_military_tension' "
            "OR EXISTS (SELECT 1 FROM gdelt_content_checked g "
            "WHERE g.event_id = c.event_id AND g.verdict = 'confirm_stage2'))"
        )
        try:
            con.execute("SELECT 1 FROM gdelt_content_checked LIMIT 1")
        except sqlite3.OperationalError:
            # labeling/run_gdelt_reclassify.py has never been run in this DB --
            # nothing has been confirmed yet, so the gate can't be evaluated.
            # Falling back to no gate here (rather than excluding everything)
            # matches this function's pre-2026-08-16 behavior for a fresh DB;
            # once the table exists, the gate is always applied.
            gate = ""

        sql = f"""
            SELECT c.event_id, c.label_source, c.event_type, c.severity_score,
                   c.confidence, c.polarity, c.countries, r.published_at
            FROM classified_events c
            JOIN raw_events r ON r.id = c.event_id
            WHERE c.event_type != 'unclassified'
            {gate}
        """
        params: list = []
        if label_sources:
            placeholders = ",".join("?" for _ in label_sources)
            sql += f" AND c.label_source IN ({placeholders})"
            params.extend(label_sources)
        if since is not None:
            sql += " AND r.published_at >= ?"
            params.append(since.isoformat())

        rows = con.execute(sql, params).fetchall()
    finally:
        con.close()

    events = []
    for row in rows:
        events.append({
            "event_id": row["event_id"],
            "label_source": row["label_source"],
            "event_type": row["event_type"],
            "severity_score": row["severity_score"],
            "confidence": row["confidence"],
            "polarity": row["polarity"],
            "countries": json.loads(row["countries"] or "[]"),
            "published_at": _parse_iso(row["published_at"]),
        })
    return events
