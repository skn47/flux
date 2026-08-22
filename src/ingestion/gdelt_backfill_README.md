# GDELT historical backfill (Phase A -> Phase F enablement)

Backfills real historical GDELT events into the same `data/events.db`
`raw_events` table the live connector writes to, so Phase E's `flux_score`
has enough historical depth to be a usable LSTM feature (Phase F). Fetches
*real* history from GDELT's own archive -- it fabricates nothing.

## Mechanism chosen, and why (verified, not assumed)

The live connector (`ingestion/gdelt.py`) polls GDELT **2.0**'s 15-minute
export (96 files/day). Reconstructing a multi-year window from those would be
~70k files for two years -- impractical.

Instead, the backfill uses GDELT **1.0**'s **daily** event export:

    http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip

Verified live on 2026-07-19 against GDELT's own file index
(`http://data.gdeltproject.org/events/index.html`): daily files exist from
**20130401** through the previous day (`20260718` present at time of writing).
This is a current, still-updated feed, not a frozen archive -- one file per
day, ~3-7 MB zipped each.

### Schema difference (verified against a real file, not memory)

The 1.0 daily export is **58 tab-separated columns**, NOT 2.0's 61. Confirmed
by downloading `20260715.export.CSV.zip` on 2026-07-19 and counting fields.
The columns this pipeline uses shift position but carry the same meaning; the
full mapping is documented in the module docstring of
`ingestion/gdelt_backfill.py`. Crucially, the country-code vocabularies are
identical to 2.0 (`Actor1/2CountryCode` = CAMEO 3-letter USA/TWN/KOR;
`ActionGeo_CountryCode` = FIPS 10-4 US/TW/KS), so the **same**
`config.mvp_scope` country filter and the **same** `labeling/gdelt_labeler.py`
apply unchanged.

### Timestamp fidelity caveat (do not paper over)

2.0's `DATEADDED` is `YYYYMMDDHHMMSS` (15-min precision). 1.0 daily's
`DATEADDED` is only `YYYYMMDD` -- **date precision, no time of day**. Backfilled
rows therefore set `published_at` to `00:00:00 UTC` of the DATEADDED date and
record `raw_metadata.published_at_fidelity = "date"`. Downstream code must
treat backfilled rows as day-granular. `DATEADDED` (when GDELT surfaced the
event), not `SQLDATE` (the event's reported date), is used for `published_at`,
matching the live connector's "only knew about it when GDELT added it"
discipline -- important for the Risk/Backtest auditor.

## Provenance tagging

Backfilled rows are `source="gdelt"` in the **same** `raw_events` table as live
rows (no second table). Their `raw_metadata` additionally carries:

    "backfill": true, "granularity": "daily", "gdelt_version": "1.0",
    "published_at_fidelity": "date"

so live (2.0, datetime-precise) vs. backfilled (1.0, date-precise) provenance
is always distinguishable downstream.

## Idempotency & failure handling

- Dedup id = `ingestion.schema.make_event_id` (normalized SOURCEURL, else
  GLOBALEVENTID) -> `INSERT OR IGNORE`. Re-running any date range never
  duplicates rows; a URL shared with a live 2.0 row dedups across versions.
- Per-day isolation: a missing daily file (GDELT occasionally drops one) or a
  network error for one date is recorded as a **gap** and does not abort the
  other dates. Gaps are reported explicitly, never silently skipped.
- Downloads are lightly parallel (default `--workers`), each with retry +
  exponential backoff; DB writes are serialized in the main thread.
- Insert count is derived from `sqlite3` `total_changes` (O(1)), NOT
  `SELECT COUNT(*)` -- the latter is O(n) per call and turns a bulk backfill
  quadratic as the table grows to millions of rows.

## How to run / re-run / extend

```
./.venv/bin/python -m ingestion.gdelt_backfill --start 2024-07-19 --end 2026-07-18 --workers 8
```

- Safe to re-run at any time (idempotent). To **extend further back**, lower
  `--start` (earliest daily file is 2013-04-01). To **catch up to today**, run
  from the last backfilled date to yesterday; the live 2.0 connector then
  covers the current day at 15-min precision.
- After ingesting, label the new rows (the standard `run_label` loads all
  unlabeled rows into memory, which OOMs on millions of rows on a small box):

```
./.venv/bin/python -m labeling.label_gdelt_backfill
```

  This calls the SAME `gdelt_labeler` and `LabelStore.insert_labels`, just
  streamed in rowid-keyed batches. Idempotent.

## Volume note (real, measured)

The MVP country filter matches any row with a USA/TWN/KOR actor -- ~33k of the
~114k rows in a daily file (USA dominates). But cross-day URL re-mentions
dedup heavily: a recent 10-day slice matched 266,535 rows and inserted only
74,540 unique (~3.6x dedup). Only QuadClass 3/4 (Verbal/Material Conflict)
rows survive `flux_engine/query.py`'s `event_type != 'unclassified'` filter
and actually feed `flux_score`; the cooperative-class rows are still ingested
and labeled (as `unclassified`) to stay faithful to the live connector's
filter, but do not drive the flux series.

## Real results (verified against the live DB, 2026-07-21) -- FINAL, CORRECTED TWICE

Two things went wrong on the way to this final state, both caught by direct verification rather
than trusting a script's own printed summary -- worth keeping in the history rather than editing
away:

1. **A false "complete" claim.** An earlier version of this section asserted a clean
   `2024-02-06 -> 2026-07-19` window. That was wrong -- written from an assumption, not a
   verification. The `2024-02-06` date actually belonged to an unrelated `source='rss'` row; the
   real GDELT coverage at that point stopped at 2025-02-09, leaving ~17 months silently missing.
   Caught by an empirical sanity check of `flux_engine` output at several historical dates (see
   `progress/flux_score_timeseries_findings.md`).
2. **A real memory leak in the gap-fill run itself**, twice observed as RSS climbing into
   multiple GB during a long run (once by you, manually killing it; once by an automated 2.9GB
   safety-valve watchdog). Root cause: `backfill()` originally submitted **every day's fetch as a
   future upfront** into a dict keyed by future, and downloads (I/O-bound) completed faster than
   the single-threaded parse+insert step could drain them, so the backlog of *finished-but-not-
   yet-consumed* results (each a full day's list of parsed rows) grew with the whole date range,
   not with `--workers`. Fixed by switching `backfill()` to a bounded pipeline that keeps at most
   `workers * 2` futures in flight at a time, submitting a replacement only as each one is
   consumed -- verified by a controlled before/after RSS trace on the same 31-day range (peak
   ~1.9GB before the fix, ~1.1GB after, and critically *plateauing* instead of still climbing).
   See the `_submit_next()` / bounded `in_flight` dict in `ingestion/gdelt_backfill.py`.

Runs executed, in order:
```
./.venv/bin/python -m ingestion.gdelt_backfill --start 2024-07-19 --end 2026-07-18 --workers 8   # first run (pre-session), stopped short at 2025-02-09
./.venv/bin/python -m ingestion.gdelt_backfill --start 2025-02-10 --end 2026-07-19 --workers 4    # gap-fill, post memory-leak fix -- succeeded, peak RSS ~1.2GB
./.venv/bin/python -m ingestion.gdelt_backfill --start 2025-02-05 --end 2025-02-07 --workers 4    # small residual gap the first run had also missed
./.venv/bin/python -m labeling.label_gdelt_backfill                                                # labeled everything newly inserted, twice (once per gap-fill)
```

**Final verified state** (direct SQL queries against `data/events.db`, not script self-reports):
- `raw_events` (`source='gdelt'`): **6,382,916** rows total.
- Daily-granularity completeness check across 2024-07-19 -> 2026-07-19 (731 calendar days): all
  covered **except 18 genuine gaps**, 2025-06-14 through 2025-07-01 inclusive -- confirmed (not
  assumed) to be missing from GDELT's own server (`404` on direct request), not a pipeline bug.
- `classified_events` (`label_source='gdelt_derived'`): **6,382,916** -- exactly matches total
  GDELT raw rows, i.e. every row, live and backfilled, has a label. Breakdown by `event_type`:

| event_type | count |
|---|---|
| `geopolitical_military_tension` (QuadClass 3/4) | 1,642,410 |
| `unclassified` (QuadClass 1/2, cooperative) | 4,740,506 |

This matches `labeling/gdelt_labeler.py`'s deterministic QuadClass rule exactly (no drift). The
`flux_engine`'s SQL-level `event_type != 'unclassified'` filter means the ~4.74M cooperative rows
are stored (for completeness/future use) but never contribute to `flux_score` -- only the ~1.64M
conflict-class rows are candidate contributors, further narrowed by the exposure graph's
country/event-type/reliability gates in `flux_engine/formula.py`.

`data/events.db` is now ~12GB after the full backfill. The stale `events.db-journal` present at
the start of this session was benign SQLite crash-recovery state (from the interrupted prior
session) and cleared itself on the first write in this run, as expected -- no corruption, no data
loss. A DB this size is worth keeping in mind for Phase F's next steps (in-memory event lists for
the flux time series should stream/batch, not load everything at once -- same discipline as
`labeling/label_gdelt_backfill.py` already applies).
