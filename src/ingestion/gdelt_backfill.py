"""
GDELT 1.0 *daily* Events export backfill connector.

Why a separate connector from ingestion/gdelt.py (the live 2.0 one)
-------------------------------------------------------------------
The live connector polls GDELT 2.0's 15-minute export
(data.gdeltproject.org/gdeltv2/lastupdate.txt). Reconstructing a long
historical window from those files means 96 files/day -- ~70k files for two
years -- which is impractical. GDELT ALSO publishes a coarser **1.0 daily**
event export, one file per day, that is still live and updated daily:

    http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip

Verified live on 2026-07-19 against GDELT's own file index
(http://data.gdeltproject.org/events/index.html): daily files exist from
20130401 through the previous day (20260718 present at time of writing), i.e.
this mechanism is current, not a frozen archive.

Schema difference from 2.0 -- verified, not assumed
---------------------------------------------------
The 1.0 daily export is **58 tab-separated columns**, NOT the 61 of 2.0.
Confirmed by downloading 20260715.export.CSV.zip on 2026-07-19 and counting
fields, cross-checked against GDELT's 1.0 CSV.header layout. The columns this
pipeline actually uses map as follows (0-based index):

    2.0 field              2.0 col   1.0 col   note
    GLOBALEVENTID          0         0
    SQLDATE                1         1
    Actor1Code             5         5
    Actor1Name             6         6
    Actor1CountryCode      7         7         CAMEO 3-letter (USA/TWN/KOR)
    Actor2Code             15        15
    Actor2Name             16        16
    Actor2CountryCode      17        17        CAMEO 3-letter
    IsRootEvent            25        25
    EventCode              26        26
    EventBaseCode          27        27
    EventRootCode          28        28
    QuadClass              29        29
    GoldsteinScale         30        30
    NumMentions            31        31
    NumSources             32        32
    NumArticles            33        33
    AvgTone                34        34
    ActionGeo_CountryCode  53        51        FIPS 2-letter (US/TW/KS)
    DATEADDED              59        56        <-- see fidelity note below
    SOURCEURL             60        57

The actor/action country vocabularies are IDENTICAL to 2.0 (CAMEO 3-letter for
Actor*CountryCode, FIPS 10-4 2-letter for *Geo_CountryCode), so the same
config.mvp_scope filter and the same gdelt_labeler apply unchanged.

TIMESTAMP FIDELITY CAVEAT (important, do not paper over)
-------------------------------------------------------
In the 2.0 export, DATEADDED is YYYYMMDDHHMMSS (15-minute precision). In the
1.0 *daily* export, DATEADDED is only YYYYMMDD -- date precision, no time of
day. This connector therefore sets `published_at` to 00:00:00 UTC of the
DATEADDED date and records `"published_at_fidelity": "date"` in raw_metadata.
Downstream code that needs sub-day precision must treat backfilled rows as
day-granular. This is a genuine loss of precision inherent to the 1.0 daily
files, not something we can recover -- it is recorded explicitly rather than
claimed away. DATEADDED (when GDELT surfaced the event), not SQLDATE (the
event's reported date), is used for published_at, matching the live 2.0
connector's discipline: the system only ever "knew" about a row on the day
GDELT added it.

Provenance tagging
------------------
Every row stores source="gdelt" in the SAME raw_events table as the live
connector (no second table), but its raw_metadata carries:
    "backfill": true, "granularity": "daily", "gdelt_version": "1.0"
so downstream code can distinguish live (2.0, datetime-precise) from
backfilled (1.0, date-precise) provenance if it ever matters.

Idempotency & failure handling
------------------------------
- Dedup id via ingestion.schema.make_event_id (normalized SOURCEURL, else
  GLOBALEVENTID) -> INSERT OR IGNORE. Re-running any date range never
  duplicates rows, and a URL shared with a live 2.0 row dedups across
  versions.
- Per-day isolation: a missing file (GDELT occasionally drops a day) or a
  network error for one date is recorded as a gap and does NOT abort the other
  dates. Gaps are reported, never silently skipped.
- Downloads are lightly parallel (default 5 workers) with retry/backoff; DB
  writes are serialized in the main thread (single sqlite connection).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import requests

from config.mvp_scope import CAMEO_COUNTRY_CODES
from src.ingestion.schema import RawEvent, make_event_id
from src.ingestion.storage import EventStore

logger = logging.getLogger("ingestion.gdelt_backfill")

SOURCE_NAME = "gdelt"
DAILY_BASE_URL = "http://data.gdeltproject.org/events"
FIRST_DAILY_DATE = date(2013, 4, 1)  # earliest daily 1.0 file, verified in index
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds; sleep = base * 2**attempt

# GDELT 1.0 daily Events column indices (0-based), 58 columns total.
COL_GLOBALEVENTID = 0
COL_SQLDATE = 1
COL_ACTOR1CODE = 5
COL_ACTOR1NAME = 6
COL_ACTOR1COUNTRYCODE = 7
COL_ACTOR2CODE = 15
COL_ACTOR2NAME = 16
COL_ACTOR2COUNTRYCODE = 17
COL_ISROOTEVENT = 25
COL_EVENTCODE = 26
COL_EVENTBASECODE = 27
COL_EVENTROOTCODE = 28
COL_QUADCLASS = 29
COL_GOLDSTEINSCALE = 30
COL_NUMMENTIONS = 31
COL_NUMSOURCES = 32
COL_NUMARTICLES = 33
COL_AVGTONE = 34
COL_ACTIONGEO_COUNTRYCODE = 51
COL_DATEADDED = 56
COL_SOURCEURL = 57
EXPECTED_NUM_COLUMNS = 58

USER_AGENT = "flux-ingest/0.1 (Phase A GDELT 1.0 daily backfill)"


class GdeltBackfillError(Exception):
    pass


@dataclass
class DayResult:
    day: date
    status: str            # "ok" | "missing" | "error"
    matched: int = 0       # rows passing the country filter (pre-dedup)
    inserted: int = 0      # newly inserted into raw_events (post-dedup)
    error: Optional[str] = None


@dataclass
class BackfillSummary:
    start: date
    end: date
    days_total: int = 0
    days_ok: int = 0
    days_missing: int = 0
    days_error: int = 0
    matched_total: int = 0
    inserted_total: int = 0
    gaps: list = field(default_factory=list)          # list[date] missing files
    errors: list = field(default_factory=list)        # list[(date, error)]
    elapsed_seconds: float = 0.0


def _daily_url(day: date) -> str:
    return f"{DAILY_BASE_URL}/{day.strftime('%Y%m%d')}.export.CSV.zip"


def _parse_dateadded_to_iso(dateadded: str) -> Optional[str]:
    """1.0 daily DATEADDED is YYYYMMDD (date only). Return ISO 8601 UTC at
    00:00:00, or None if unparseable."""
    try:
        dt = datetime.strptime(dateadded.strip(), "%Y%m%d").replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except (ValueError, AttributeError):
        return None


def _row_matches_mvp_countries(row: list[str]) -> bool:
    a1 = row[COL_ACTOR1COUNTRYCODE].strip()
    a2 = row[COL_ACTOR2COUNTRYCODE].strip()
    return a1 in CAMEO_COUNTRY_CODES or a2 in CAMEO_COUNTRY_CODES


def _row_to_raw_event(row: list[str], day: date, ingested_at: str) -> Optional[RawEvent]:
    global_event_id = row[COL_GLOBALEVENTID].strip() or None
    dateadded_raw = row[COL_DATEADDED].strip()
    published_at = _parse_dateadded_to_iso(dateadded_raw)
    published_at_fallback = None
    if published_at is None:
        # Fall back to the file's own date rather than fabricating a time.
        published_at = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).isoformat()
        published_at_fallback = "file_date"

    sourceurl_raw = row[COL_SOURCEURL].strip()
    url = sourceurl_raw if sourceurl_raw.lower().startswith("http") else None

    try:
        event_id = make_event_id(SOURCE_NAME, url, global_event_id)
    except ValueError:
        return None

    def _f(idx: int) -> Optional[float]:
        v = row[idx].strip()
        try:
            return float(v) if v else None
        except ValueError:
            return None

    def _i(idx: int) -> Optional[int]:
        v = row[idx].strip()
        try:
            return int(v) if v else None
        except ValueError:
            return None

    raw_metadata = {
        "GLOBALEVENTID": global_event_id,
        "SQLDATE": row[COL_SQLDATE].strip() or None,
        "Actor1Code": row[COL_ACTOR1CODE].strip() or None,
        "Actor1Name": row[COL_ACTOR1NAME].strip() or None,
        "Actor1CountryCode": row[COL_ACTOR1COUNTRYCODE].strip() or None,
        "Actor2Code": row[COL_ACTOR2CODE].strip() or None,
        "Actor2Name": row[COL_ACTOR2NAME].strip() or None,
        "Actor2CountryCode": row[COL_ACTOR2COUNTRYCODE].strip() or None,
        "IsRootEvent": row[COL_ISROOTEVENT].strip() or None,
        "EventCode": row[COL_EVENTCODE].strip() or None,
        "EventBaseCode": row[COL_EVENTBASECODE].strip() or None,
        "EventRootCode": row[COL_EVENTROOTCODE].strip() or None,
        "QuadClass": _i(COL_QUADCLASS),
        "GoldsteinScale": _f(COL_GOLDSTEINSCALE),
        "NumMentions": _i(COL_NUMMENTIONS),
        "NumSources": _i(COL_NUMSOURCES),
        "NumArticles": _i(COL_NUMARTICLES),
        "AvgTone": _f(COL_AVGTONE),
        "ActionGeo_CountryCode": row[COL_ACTIONGEO_COUNTRYCODE].strip() or None,
        "SOURCEURL": sourceurl_raw or None,
        "DATEADDED": dateadded_raw or None,
        # --- backfill provenance / fidelity markers ---
        "backfill": True,
        "granularity": "daily",
        "gdelt_version": "1.0",
        "published_at_fidelity": "date",  # date-only precision, see module docstring
    }
    if published_at_fallback:
        raw_metadata["published_at_fallback"] = published_at_fallback

    return RawEvent(
        id=event_id,
        source=SOURCE_NAME,
        source_id=global_event_id,
        published_at=published_at,
        ingested_at=ingested_at,
        title=None,
        text=None,
        url=url,
        raw_metadata=raw_metadata,
    )


_INSERT_SQL = """
INSERT OR IGNORE INTO raw_events
    (id, source, source_id, published_at, ingested_at, title, text, url, raw_metadata)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _insert_events_fast(conn, events: list[RawEvent]) -> int:
    """Insert with the SAME INSERT OR IGNORE + schema as
    ingestion.storage.EventStore.insert_events, but derive the newly-inserted
    count from `conn.total_changes` (O(1)) instead of two `SELECT COUNT(*)`
    scans. Over a multi-million-row table those COUNT(*) calls are O(n) and,
    called twice per day across a 2-year backfill, become quadratic -- the
    single biggest speed bug in a naive bulk backfill. Idempotency is
    unchanged (still INSERT OR IGNORE on the `id` primary key)."""
    if not events:
        return 0
    rows = [
        (
            e.id, e.source, e.source_id, e.published_at, e.ingested_at,
            e.title, e.text, e.url,
            json.dumps(e.raw_metadata, ensure_ascii=False, default=str),
        )
        for e in events
    ]
    before = conn.total_changes
    conn.executemany(_INSERT_SQL, rows)
    conn.commit()
    return conn.total_changes - before


def _fetch_day_events(
    session: requests.Session, day: date, ingested_at: str
) -> tuple[str, list[RawEvent], Optional[str]]:
    """Download+parse one daily file. Returns (status, events, error).
    status is "ok" | "missing" | "error". Never raises."""
    url = _daily_url(day)
    last_error: Optional[str] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 404:
                return ("missing", [], f"404 for {url}")
            resp.raise_for_status()
            events: list[RawEvent] = []
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                names = zf.namelist()
                if not names:
                    return ("error", [], f"empty zip {url}")
                with zf.open(names[0]) as f:
                    stream = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                    reader = csv.reader(stream, delimiter="\t")
                    for row in reader:
                        if len(row) != EXPECTED_NUM_COLUMNS:
                            continue
                        if not _row_matches_mvp_countries(row):
                            continue
                        ev = _row_to_raw_event(row, day, ingested_at)
                        if ev is not None:
                            events.append(ev)
            return ("ok", events, None)
        except Exception as exc:  # noqa: BLE001 - retry then give up gracefully
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
    return ("error", [], last_error)


def backfill(
    start: date,
    end: date,
    workers: int = 5,
    db_path=None,
    progress_every: int = 20,
) -> BackfillSummary:
    """Backfill GDELT 1.0 daily exports for [start, end] inclusive into
    raw_events. Idempotent. Downloads lightly in parallel; inserts serially."""
    if start < FIRST_DAILY_DATE:
        raise GdeltBackfillError(
            f"start {start} predates earliest daily file {FIRST_DAILY_DATE}"
        )
    if end < start:
        raise GdeltBackfillError(f"end {end} before start {start}")

    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    summary = BackfillSummary(start=start, end=end, days_total=len(days))
    t0 = time.time()
    ingested_at = datetime.now(timezone.utc).isoformat()

    store = EventStore() if db_path is None else EventStore(db_path)
    # Per-connection bulk-insert tuning (does not persist to the file the way
    # journal_mode would): trade a little crash-durability for speed. The
    # backfill is idempotent, so a crash just means re-running to fill gaps.
    store.conn.execute("PRAGMA synchronous=NORMAL")
    store.conn.execute("PRAGMA cache_size=-100000")  # ~100 MB page cache
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    processed = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Bounded pipeline, NOT "submit every day upfront". The original
            # version built a dict of ALL 525 futures before consuming any of
            # them; downloads (I/O-bound, `workers` of them at a time) kept
            # completing and queuing up inside as_completed() faster than the
            # single-threaded parse-and-insert step could drain them, so the
            # count of *finished-but-not-yet-consumed* results (each a full
            # day's list of RawEvent objects, tens of thousands of rows) grew
            # roughly with the whole date range, not with `workers` -- this
            # was the real cause of RSS climbing into multiple GB on a
            # 525-day run, confirmed by a rolling-window RSS trace showing
            # memory tracking the size of the unconsumed backlog, not a
            # per-future reference leak. Capping in-flight futures at a small
            # multiple of `workers` bounds peak memory to a constant
            # independent of total days, regardless of any download/consume
            # rate mismatch.
            max_in_flight = max(workers * 2, 1)
            days_iter = iter(days)
            in_flight: dict = {}

            def _submit_next() -> bool:
                try:
                    day = next(days_iter)
                except StopIteration:
                    return False
                in_flight[pool.submit(_fetch_day_events, session, day, ingested_at)] = day
                return True

            for _ in range(min(max_in_flight, len(days))):
                _submit_next()

            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    day = in_flight.pop(fut)
                    status, events, error = fut.result()
                    if status == "ok":
                        matched = len(events)
                        inserted = _insert_events_fast(store.conn, events)
                        summary.days_ok += 1
                        summary.matched_total += matched
                        summary.inserted_total += inserted
                    elif status == "missing":
                        summary.days_missing += 1
                        summary.gaps.append(day)
                        logger.warning("gdelt_backfill: MISSING file for %s (%s)", day, error)
                    else:
                        summary.days_error += 1
                        summary.errors.append((day, error))
                        logger.error("gdelt_backfill: ERROR for %s: %s", day, error)

                    processed += 1
                    if processed % progress_every == 0:
                        logger.info(
                            "gdelt_backfill: %d/%d days processed, %d rows inserted so far",
                            processed, summary.days_total, summary.inserted_total,
                        )
                    _submit_next()
    finally:
        store.close()

    summary.gaps.sort()
    summary.elapsed_seconds = time.time() - t0
    return summary


def _print_summary(s: BackfillSummary) -> None:
    print("\n=== GDELT 1.0 daily backfill summary ===")
    print(f"window:        {s.start} .. {s.end}  ({s.days_total} days)")
    print(f"days ok:       {s.days_ok}")
    print(f"days missing:  {s.days_missing}")
    print(f"days error:    {s.days_error}")
    print(f"rows matched (country filter, pre-dedup): {s.matched_total}")
    print(f"rows inserted (post-dedup, new):          {s.inserted_total}")
    print(f"elapsed:       {s.elapsed_seconds:.1f}s")
    if s.gaps:
        print(f"\nGAPS (missing daily files, {len(s.gaps)}):")
        for d in s.gaps:
            print(f"  {d}")
    if s.errors:
        print(f"\nERRORS ({len(s.errors)}):")
        for d, e in s.errors:
            print(f"  {d}: {e}")
    print()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="GDELT 1.0 daily events backfill")
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--workers", type=int, default=5)
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    summary = backfill(start, end, workers=args.workers)
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
