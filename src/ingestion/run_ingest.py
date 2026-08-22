"""
Phase A ingestion orchestrator.

Run with:
    python -m ingestion.run_ingest

Runs every connector (gdelt, rss, newsapi), catching and logging exceptions
per-source so one broken source can never take down the others. Prints a
summary table (rows fetched / newly inserted / deduped-away / error) and
exits non-zero only if every single source failed outright.

A connector "failing" means it raised an exception (network error, bad zip,
unparseable response, etc). NewsAPI cleanly skipping because NEWSAPI_KEY is
unset is NOT a failure -- it's an expected, logged no-op (0 fetched, 0
inserted), and does not count against the "every source failed" exit check.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

from src.ingestion import gdelt, newsapi, rss
from src.ingestion.env import load_env_file
from src.ingestion.storage import EventStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("ingestion.run_ingest")


@dataclass
class SourceResult:
    source: str
    fetched: int = 0
    inserted: int = 0
    deduped: int = 0
    error: str | None = None


def run_source(name: str, fetch_fn, store: EventStore) -> SourceResult:
    result = SourceResult(source=name)
    try:
        events = fetch_fn()
    except Exception as exc:  # noqa: BLE001 - a single broken source must not kill the run
        logger.error("%s: connector raised an exception: %s", name, exc, exc_info=True)
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    result.fetched = len(events)
    if events:
        attempted, inserted = store.insert_events(events)
        result.inserted = inserted
        result.deduped = attempted - inserted
    return result


def print_summary(results: list[SourceResult]) -> None:
    print("\n=== Ingestion summary ===")
    header = f"{'source':10} {'fetched':>8} {'inserted':>9} {'deduped':>8}  error"
    print(header)
    print("-" * len(header))
    for r in results:
        err = r.error or ""
        print(f"{r.source:10} {r.fetched:8d} {r.inserted:9d} {r.deduped:8d}  {err}")
    print()


def main() -> int:
    load_env_file()  # populate os.environ from .env if present, before any connector runs

    store = EventStore()
    sources = [
        ("gdelt", gdelt.fetch),
        ("rss", rss.fetch),
        ("newsapi", newsapi.fetch),
    ]

    results = [run_source(name, fn, store) for name, fn in sources]
    store.close()

    print_summary(results)

    all_failed = all(r.error is not None for r in results)
    if all_failed:
        logger.error("Every source failed -- exiting non-zero.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
