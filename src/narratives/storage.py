"""
SQLite-backed store for AI-generated per-event causal narratives, added
2026-08-16 for the 3D causality-graph feature (see labeling/README.md and
frontend/src/components/CausalityScene3D.tsx).

Adds a single table `event_narratives` to the same `data/events.db` file
used everywhere else (`ingestion.storage.DEFAULT_DB_PATH`, reused, never
duplicated) -- same convention as `labeling/storage.py::LabelStore`.

Deliberately its OWN table, not a column bolted onto `event_catalog`:
`event_catalog` is rebuilt fast/synchronously by `flux_engine.run_timeseries`
on every pipeline refresh, while narrative generation is slow/LLM-bound
(~5.7s/call, see narratives/ollama_narrator.py) and run as a separate,
resumable batch job -- the same separation of concerns already established
between `flux_engine` (fast/sync) and `labeling.run_gdelt_reclassify` (slow/
LLM-bound). One row per event_id (not per event_id+ticker): a narrative
explains why the EVENT matters, grounded in its own highest-impact exposure
paths -- it is not re-generated per ticker it happens to reach.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.ingestion.storage import DEFAULT_DB_PATH
from src.narratives.ollama_narrator import EventNarrative

_SCHEMA = """
CREATE TABLE IF NOT EXISTS event_narratives (
    event_id        TEXT PRIMARY KEY,
    narrative       TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    generated_at    TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES event_catalog(event_id)
);
"""


class NarrativeStore:
    """Thin wrapper around the shared sqlite3 connection to data/events.db."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def insert_narrative(self, n: EventNarrative) -> None:
        """Commit immediately (not batched) so a Ctrl-C mid-run loses at
        most one in-flight LLM call, not the whole batch -- matches
        labeling/run_gdelt_reclassify.py's per-row commit discipline."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO event_narratives
                (event_id, narrative, model_version, generated_at)
            VALUES (?, ?, ?, ?)
            """,
            (n.event_id, n.narrative, n.model_version, n.generated_at),
        )
        self.conn.commit()

    def narrated_event_ids(self) -> set[str]:
        return {row[0] for row in self.conn.execute("SELECT event_id FROM event_narratives")}

    def count_rows(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM event_narratives").fetchone()[0]

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "NarrativeStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
