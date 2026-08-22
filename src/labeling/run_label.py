"""
Phase B labeling orchestrator.

Run with:
    ./.venv/bin/python -m labeling.run_label

For each labeler, queries `classified_events` for the not-yet-labeled-by-this-
labeler rows first (LabelStore.unlabeled_raw_events), so a second run does
zero redundant *computation*, not just zero redundant inserts:
  - source == 'gdelt'          -> gdelt_labeler   (label_source=gdelt_derived)
  - source in ('rss','newsapi') -> rule_labeler    (label_source=rule_based)
  - source in ('rss','newsapi') -> llm_labeler     (label_source=llm_assisted),
    only attempted if ANTHROPIC_API_KEY is present (checked once up front so
    the summary can say "skipped" rather than silently doing nothing)

Catches/logs exceptions per-labeler, same discipline as
`ingestion/run_ingest.py`'s per-source handling -- one broken labeler must
never kill the run for the others. Prints a summary: rows newly labeled this
run per labeler, plus a whole-corpus breakdown by event_type/severity and a
count of confidence < 0.3 ("zero confident label") rows.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from dataclasses import dataclass

from src.ingestion.env import load_env_file
from src.labeling import gdelt_labeler, llm_labeler, rule_labeler
from src.labeling.schema import SEVERITY_NAMES
from src.labeling.storage import LabelStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("labeling.run_label")

TEXT_SOURCES = ("rss", "newsapi")


@dataclass
class LabelerResult:
    labeler: str
    label_source: str
    candidates: int = 0     # rows fetched as not-yet-labeled
    labeled: int = 0        # ClassifiedEvent objects produced
    inserted: int = 0       # newly inserted (idempotency-aware)
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


def _already_content_checked_ids(store: LabelStore) -> set[str]:
    """
    Event ids `labeling/run_gdelt_reclassify.py` has already put through its
    content-based correction pass (`gdelt_content_checked`).

    Bug found 2026-08-16 (see labeling/README.md): `unlabeled_raw_events`
    below is a pure anti-join against `classified_events`. A "reject" verdict
    in run_gdelt_reclassify.py deletes the classified_events row (its
    delete-only correction semantics), which makes that row look "unlabeled"
    again from this anti-join's point of view -- so a later, even unrelated,
    `run_label` invocation would silently re-insert it with the original,
    uncorrected QuadClass-only label, undoing the correction. Worse,
    run_gdelt_reclassify.py's own candidate query excludes anything already in
    `gdelt_content_checked`, so the resurrected wrong label is never re-caught
    either -- it's permanently stuck. One `run_label` run (for an unrelated
    rule_labeler.py fix) resurrected all 9,528 previously-corrected false
    positives this way, 3,408 of them live cluster winners in event_catalog.
    Excluding every checked id here (confirm or reject) closes the gap:
    confirmed rows were never deleted so this is a no-op for them, and
    rejected rows now correctly stay gone instead of coming back.
    """
    try:
        rows = store.conn.execute("SELECT event_id FROM gdelt_content_checked").fetchall()
    except sqlite3.OperationalError:
        return set()  # run_gdelt_reclassify.py has never been run -- nothing to exclude
    return {row[0] for row in rows}


def _run_gdelt(store: LabelStore) -> LabelerResult:
    result = LabelerResult(labeler="gdelt_labeler", label_source="gdelt_derived")
    try:
        candidates = store.unlabeled_raw_events(source="gdelt", label_source="gdelt_derived")
        checked = _already_content_checked_ids(store)
        candidates = [c for c in candidates if c["id"] not in checked]
        result.candidates = len(candidates)
        events = gdelt_labeler.label_batch(candidates)
        result.labeled = len(events)
        if events:
            attempted, inserted = store.insert_labels(events)
            result.inserted = inserted
    except Exception as exc:  # noqa: BLE001 - one broken labeler must not kill the run
        logger.error("gdelt_labeler: raised an exception: %s", exc, exc_info=True)
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def _run_rule(store: LabelStore) -> list[LabelerResult]:
    results = []
    for source in TEXT_SOURCES:
        result = LabelerResult(labeler=f"rule_labeler[{source}]", label_source="rule_based")
        try:
            candidates = store.unlabeled_raw_events(source=source, label_source="rule_based")
            result.candidates = len(candidates)
            events = rule_labeler.label_batch(candidates)
            result.labeled = len(events)
            if events:
                attempted, inserted = store.insert_labels(events)
                result.inserted = inserted
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "rule_labeler[%s]: raised an exception: %s", source, exc, exc_info=True
            )
            result.error = f"{type(exc).__name__}: {exc}"
        results.append(result)
    return results


def _run_llm(store: LabelStore) -> list[LabelerResult]:
    results = []
    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    for source in TEXT_SOURCES:
        result = LabelerResult(labeler=f"llm_labeler[{source}]", label_source="llm_assisted")
        if not have_key:
            result.skipped = True
            result.skip_reason = "ANTHROPIC_API_KEY not set"
            results.append(result)
            continue
        try:
            candidates = store.unlabeled_raw_events(source=source, label_source="llm_assisted")
            result.candidates = len(candidates)
            events = llm_labeler.label_batch(candidates)
            result.labeled = len(events)
            if events:
                attempted, inserted = store.insert_labels(events)
                result.inserted = inserted
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "llm_labeler[%s]: raised an exception: %s", source, exc, exc_info=True
            )
            result.error = f"{type(exc).__name__}: {exc}"
        results.append(result)
    return results


def print_run_summary(results: list[LabelerResult]) -> None:
    print("\n=== Labeling run summary (this run) ===")
    header = f"{'labeler':22} {'candidates':>10} {'labeled':>8} {'inserted':>9}  status"
    print(header)
    print("-" * len(header))
    for r in results:
        if r.skipped:
            status = f"skipped ({r.skip_reason})"
        elif r.error:
            status = f"ERROR: {r.error}"
        else:
            status = "ok"
        print(f"{r.labeler:22} {r.candidates:10d} {r.labeled:8d} {r.inserted:9d}  {status}")
    print()


def print_corpus_summary(store: LabelStore) -> None:
    summary = store.summary_counts()
    print("=== classified_events corpus summary (whole table, all runs) ===")
    print(f"total rows: {summary['total']}")

    print("\nby label_source:")
    for label_source, n in sorted(summary["by_label_source"].items()):
        print(f"  {label_source:15} {n}")

    print("\nby event_type:")
    for event_type, n in sorted(summary["by_event_type"].items(), key=lambda kv: -kv[1]):
        print(f"  {event_type:32} {n}")

    print("\nby severity:")
    for severity in sorted(summary["by_severity"].keys()):
        name = SEVERITY_NAMES.get(severity, "?")
        print(f"  {severity} ({name:10}) {summary['by_severity'][severity]}")

    print(f"\nlow-confidence rows (confidence < 0.3): {summary['low_confidence_count']}")
    print()


def main() -> int:
    load_env_file()  # populate os.environ from .env if present, before any labeler runs

    store = LabelStore()

    all_results: list[LabelerResult] = []
    all_results.append(_run_gdelt(store))
    all_results.extend(_run_rule(store))
    all_results.extend(_run_llm(store))

    print_run_summary(all_results)
    print_corpus_summary(store)

    store.close()

    hard_errors = [r for r in all_results if r.error is not None]
    if len(hard_errors) == len(all_results):
        logger.error("Every labeler failed -- exiting non-zero.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
