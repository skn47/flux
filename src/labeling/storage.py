"""
SQLite-backed classification store for Phase B labeling.

Adds a single table `classified_events` to the *same* `data/events.db` file
used by Phase A ingestion (`ingestion.storage.DEFAULT_DB_PATH`, reused here,
never duplicated). One row per `(event_id, label_source)` -- the same
underlying `raw_events.id` can carry independent labels from multiple
labelers (gdelt_derived / rule_based / llm_assisted / manual) without
collision, and idempotency is enforced the same way Phase A does it:
`INSERT OR IGNORE` keyed on the primary key.

Operational note (also in labeling/README.md): relabeling after editing a
keyword dictionary or the taxonomy requires deleting that label_source's rows
first, e.g.:
    DELETE FROM classified_events WHERE label_source = 'rule_based';
This is deliberately not automated, to avoid silently discarding prior labels.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from src.ingestion.storage import DEFAULT_DB_PATH
from src.labeling.schema import ClassifiedEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS classified_events (
    event_id        TEXT NOT NULL,
    label_source    TEXT NOT NULL,   -- gdelt_derived | rule_based | llm_assisted | manual
    event_type      TEXT NOT NULL,
    severity        INTEGER NOT NULL,
    severity_score  REAL NOT NULL,
    polarity        REAL,
    countries       TEXT NOT NULL DEFAULT '[]',   -- JSON list, mirrors raw_metadata's JSON-text pattern
    sector          TEXT,
    companies       TEXT NOT NULL DEFAULT '[]',   -- JSON list of tickers
    confidence      REAL NOT NULL,
    reasoning       TEXT,
    labeler_version TEXT,
    labeled_at      TEXT NOT NULL,
    PRIMARY KEY (event_id, label_source),
    FOREIGN KEY (event_id) REFERENCES raw_events(id)
);
CREATE INDEX IF NOT EXISTS idx_classified_events_event_type ON classified_events(event_type);
CREATE INDEX IF NOT EXISTS idx_classified_events_label_source ON classified_events(label_source);
"""


class LabelStore:
    """Thin wrapper around the shared sqlite3 connection to data/events.db."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- writes ---------------------------------------------------------

    def insert_labels(self, events: Iterable[ClassifiedEvent]) -> tuple[int, int]:
        """
        Insert classified events, ignoring any whose (event_id, label_source)
        already exists. Returns (attempted, newly_inserted).
        """
        rows = [
            (
                e.event_id,
                e.label_source,
                e.event_type,
                e.severity,
                e.severity_score,
                e.polarity,
                json.dumps(e.countries, ensure_ascii=False),
                e.sector,
                json.dumps(e.companies, ensure_ascii=False),
                e.confidence,
                e.reasoning,
                e.labeler_version,
                e.labeled_at,
            )
            for e in events
        ]
        if not rows:
            return (0, 0)

        before = self.count_rows()
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO classified_events
                (event_id, label_source, event_type, severity, severity_score,
                 polarity, countries, sector, companies, confidence, reasoning,
                 labeler_version, labeled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()
        after = self.count_rows()
        return (len(rows), after - before)

    def delete_label(self, event_id: str, label_source: str) -> None:
        """Scoped single-row delete, for labeling/run_gdelt_reclassify.py's
        per-event corrections -- same (event_id, label_source) PK the rest of
        this class keys on. See module docstring's note on the existing bulk
        `DELETE FROM classified_events WHERE label_source = ...` idiom this
        generalizes to a single event."""
        self.conn.execute(
            "DELETE FROM classified_events WHERE event_id = ? AND label_source = ?",
            (event_id, label_source),
        )
        self.conn.commit()

    # -- reads ------------------------------------------------------------

    def unlabeled_raw_events(self, source: str, label_source: str) -> list[dict]:
        """
        Anti-join: raw_events rows of `source` that do not yet have a row in
        classified_events for `label_source`. This is what makes a second
        orchestrator run do zero redundant *computation*, not just zero
        redundant inserts (the labelers are never even called on rows that
        are already labeled by this label_source).

        Returns plain dicts (not sqlite3.Row) with `raw_metadata` already
        json.loads'd, so labelers never have to touch JSON parsing directly.
        """
        cur = self.conn.execute(
            """
            SELECT r.*
            FROM raw_events r
            LEFT JOIN classified_events c
                ON r.id = c.event_id AND c.label_source = ?
            WHERE r.source = ? AND c.event_id IS NULL
            """,
            (label_source, source),
        )
        rows = []
        for row in cur.fetchall():
            d = dict(row)
            try:
                d["raw_metadata"] = json.loads(d.get("raw_metadata") or "{}")
            except (TypeError, ValueError):
                d["raw_metadata"] = {}
            rows.append(d)
        return rows

    def count_rows(self, label_source: str | None = None) -> int:
        if label_source:
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM classified_events WHERE label_source = ?",
                (label_source,),
            )
        else:
            cur = self.conn.execute("SELECT COUNT(*) FROM classified_events")
        return cur.fetchone()[0]

    def summary_counts(self) -> dict:
        """
        Whole-corpus snapshot (not just this run) for the orchestrator's
        printed summary: counts per label_source, per event_type, per
        severity band, and a count of "zero confident" rows (confidence < 0.3).
        """
        summary: dict = {"by_label_source": {}, "by_event_type": {}, "by_severity": {}}

        for row in self.conn.execute(
            "SELECT label_source, COUNT(*) AS n FROM classified_events GROUP BY label_source"
        ):
            summary["by_label_source"][row["label_source"]] = row["n"]

        for row in self.conn.execute(
            "SELECT event_type, COUNT(*) AS n FROM classified_events GROUP BY event_type"
        ):
            summary["by_event_type"][row["event_type"]] = row["n"]

        for row in self.conn.execute(
            "SELECT severity, COUNT(*) AS n FROM classified_events GROUP BY severity"
        ):
            summary["by_severity"][row["severity"]] = row["n"]

        cur = self.conn.execute(
            "SELECT COUNT(*) FROM classified_events WHERE confidence < 0.3"
        )
        summary["low_confidence_count"] = cur.fetchone()[0]
        summary["total"] = self.count_rows()
        return summary

    def sample_labeled(
        self, label_source: str, sources: tuple[str, ...], limit: int = 20
    ) -> list[sqlite3.Row]:
        """
        Random sample of `label_source`-labeled rows, joined back to their
        raw_events title/text/url, for hand spot-checking. `sources`
        restricts to raw_events.source values (e.g. ('rss', 'newsapi')).
        """
        placeholders = ",".join("?" for _ in sources)
        cur = self.conn.execute(
            f"""
            SELECT r.title, r.text, r.url, r.source,
                   c.event_type, c.severity, c.severity_score, c.confidence,
                   c.sector, c.companies, c.countries, c.reasoning
            FROM classified_events c
            JOIN raw_events r ON r.id = c.event_id
            WHERE c.label_source = ? AND r.source IN ({placeholders})
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (label_source, *sources, limit),
        )
        return cur.fetchall()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "LabelStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
