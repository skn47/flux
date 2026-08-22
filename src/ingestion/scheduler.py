"""
Ingestion scheduler (Direction 3).

`ingestion/run_ingest.py` is a one-shot script -- someone has to run it by
hand. GDELT itself publishes a new 15-minute Events export continuously
regardless of whether anyone pulls it (see `ingestion/gdelt.py`), so
leaving ingestion manual-only means `raw_events` goes stale between
whenever a human happens to rerun the pipeline. This module closes that gap
for the ingestion stage ONLY: a small stdlib-only loop that reruns the same
gdelt/rss/newsapi cycle `ingestion/run_ingest.py::main` already runs, on a
fixed interval, forever.

Deliberately does NOT touch labeling/price/flux_engine/features/lstm/
backtest -- those stay on the existing manual `scripts/refresh_pipeline.sh`
trigger (see `api/README.md`'s "Data freshness" section). Newly ingested
rows sit in `raw_events` unused by the API until someone runs that script;
running this scheduler does not make `flux_score` or predictions any
fresher on its own.

Run with:
    ./.venv/bin/python -m ingestion.scheduler
    ./.venv/bin/python -m ingestion.scheduler --interval 1800
    ./.venv/bin/python -m ingestion.scheduler --once   # one cycle, then exit

Typically left running in the background:
    nohup ./.venv/bin/python -m ingestion.scheduler > ingestion_scheduler.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.ingestion import gdelt, newsapi, rss
from src.ingestion.env import load_env_file
from src.ingestion.run_ingest import SourceResult, print_summary, run_source
from src.ingestion.storage import EventStore

logger = logging.getLogger("ingestion.scheduler")

# Matches GDELT's own 15-minute export cadence (ingestion/gdelt.py pulls
# "the latest" export each call) -- polling faster than GDELT itself
# publishes is wasted work, polling looser delays freshness for no reason.
DEFAULT_INTERVAL_SECONDS = 900

STATUS_PATH = Path(__file__).resolve().parent.parent.parent / "ingestion" / "scheduler_status.json"

_shutdown_requested = False


def _handle_signal(signum: int, _frame) -> None:
    global _shutdown_requested
    logger.info("Received signal %s -- will exit after the current cycle.", signal.Signals(signum).name)
    _shutdown_requested = True


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cycle() -> list[SourceResult]:
    """One full ingestion cycle: same gdelt/rss/newsapi run
    `ingestion.run_ingest.main` performs, reusing its exact building blocks
    (`run_source`, `SourceResult`, `print_summary`) rather than duplicating
    or modifying that module -- only its per-source-result orchestration is
    inlined here, since `main()` itself only returns an exit code, not the
    results this scheduler needs for `scheduler_status.json`."""
    load_env_file()
    store = EventStore()
    sources = [
        ("gdelt", gdelt.fetch),
        ("rss", rss.fetch),
        ("newsapi", newsapi.fetch),
    ]
    results = [run_source(name, fn, store) for name, fn in sources]
    store.close()
    print_summary(results)
    return results


def _write_status(cycles_completed: int, results: list[SourceResult], next_run_at: str | None) -> None:
    status = {
        "last_run_at": _now_iso(),
        "last_run_results": [asdict(r) for r in results],
        "cycles_completed": cycles_completed,
        "next_run_at": next_run_at,
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2))


def run_forever(interval_seconds: float = DEFAULT_INTERVAL_SECONDS, once: bool = False) -> int:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    cycles_completed = 0
    last_results: list[SourceResult] = []
    while True:
        if _shutdown_requested:
            logger.info("Exiting after %d cycle(s) (signal received during sleep).", cycles_completed)
            return 0

        cycle_start = time.monotonic()
        logger.info("Cycle %d starting...", cycles_completed + 1)
        try:
            last_results = run_cycle()
        except Exception:  # noqa: BLE001 - a daemon meant to run for days must survive an unexpected crash
            logger.error("Ingestion cycle raised an unexpected exception -- will retry next cycle.", exc_info=True)
            last_results = []
        cycles_completed += 1

        elapsed = time.monotonic() - cycle_start
        remaining = max(0.0, interval_seconds - elapsed)
        next_run_at = None if once else (datetime.now(timezone.utc) + timedelta(seconds=remaining)).isoformat()
        _write_status(cycles_completed, last_results, next_run_at)

        all_failed = bool(last_results) and all(r.error is not None for r in last_results)
        exit_code = 1 if all_failed else 0

        if once or _shutdown_requested:
            logger.info("Exiting after %d cycle(s).", cycles_completed)
            return exit_code

        logger.info("Cycle %d done in %.1fs -- sleeping %.1fs until next cycle.", cycles_completed, elapsed, remaining)
        slept = 0.0
        while slept < remaining and not _shutdown_requested:
            chunk = min(1.0, remaining - slept)
            time.sleep(chunk)
            slept += chunk


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ingestion.run_ingest on a fixed interval, forever.")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SECONDS, help="Seconds between cycles (default: 900, matching GDELT's own export cadence).")
    parser.add_argument("--once", action="store_true", help="Run exactly one cycle and exit (for testing).")
    args = parser.parse_args()

    # ingestion.run_ingest already calls logging.basicConfig at module level
    # (imported above), so the root logger is configured as soon as this
    # module loads -- no need to configure it again here.
    return run_forever(interval_seconds=args.interval, once=args.once)


if __name__ == "__main__":
    sys.exit(main())
