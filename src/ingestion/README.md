# Ingestion (Phase A)

First stage of the financial-flux-intelligence pipeline: pulls raw
events from external feeds into a clean, deduplicated SQLite event store.
Does no classification, scoring, or sector judgment -- that's Phase B/C.
This stage's only job is to ingest faithfully and never fabricate or
silently backfill data.

## Schema

Every connector normalizes into `ingestion.schema.RawEvent`:
`id, source, source_id, published_at, ingested_at, title, text, url,
raw_metadata`. `id` is `sha256(source + "|" + normalized_url)`, falling back
to `sha256(source + "|" + source_id)` when a record has no URL. Source-specific
fields (GDELT's GoldsteinScale/AvgTone/actor codes, RSS feed name/tags, etc.)
live in `raw_metadata` as JSON -- they never leak into the shared columns.
`published_at` is always the source's own event/publish time when available;
`ingested_at` is always when this run pulled the record. If a source has no
publish timestamp, the connector falls back to `ingested_at` for
`published_at` but records `raw_metadata.published_at_fallback` so the fact
of the fallback is never hidden.

## Storage

SQLite at `data/events.db`, single table `raw_events`, primary key `id`.
Inserts use `INSERT OR IGNORE` keyed on `id`, so re-running ingestion is
idempotent by construction. `ingestion/storage.py` also has `EventStore.recent()`
for pulling recent rows to sanity-check.

## Running it

```
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m ingestion.run_ingest
```

Runs GDELT, RSS, and NewsAPI, catching exceptions per-source (one broken
source never kills the others), and prints a fetched/inserted/deduped/error
summary per source. Exit code is non-zero only if every source failed
outright.

### Running it on a schedule (Direction 3)

`ingestion/run_ingest.py` above is one-shot -- a human has to rerun it.
`ingestion/scheduler.py` reruns the exact same gdelt/rss/newsapi cycle
automatically, forever, on a fixed interval:

```
./.venv/bin/python -m ingestion.scheduler                 # default: every 900s (15 min)
./.venv/bin/python -m ingestion.scheduler --interval 1800  # custom interval, in seconds
./.venv/bin/python -m ingestion.scheduler --once           # one cycle, then exit -- for testing
```

The default 900s interval matches GDELT's own export cadence
(`ingestion/gdelt.py` always pulls "the latest" 15-minute export --
polling faster than GDELT publishes is wasted work). Typically left running
in the background:

```
nohup ./.venv/bin/python -m ingestion.scheduler > ingestion_scheduler.log 2>&1 &
```

Handles `SIGINT`/`SIGTERM` for a clean shutdown (finishes or skips the
in-flight cycle rather than being killed mid-write), and writes
`ingestion/scheduler_status.json` after every cycle
(`last_run_at`, `last_run_results` -- the same per-source
fetched/inserted/deduped/error fields as the summary table above,
`cycles_completed`, `next_run_at`) so a daemon's health can be checked
without tailing logs.

**Scope note:** this only automates ingestion. Labeling, price, flux_engine,
features, LSTM training, and backtest are still manual-only via
`scripts/refresh_pipeline.sh` -- see `api/README.md`'s "Data freshness"
section. Running this scheduler means `raw_events` stays fresh on its own;
it does **not** mean `flux_score` or predictions do.

## Source status

- **GDELT** (`ingestion/gdelt.py`) -- live, no key needed. Pulls the latest
  15-minute Events export from `data.gdeltproject.org/gdeltv2/lastupdate.txt`,
  filters to rows where `Actor1CountryCode`/`Actor2CountryCode` match the MVP
  countries. **Country code nuance, verified against GDELT's own reference
  files (not memory):** GDELT uses two vocabularies in the same table --
  `Actor1CountryCode`/`Actor2CountryCode` are 3-letter **CAMEO** codes
  (`USA`, `TWN`, `KOR`; see
  `https://www.gdeltproject.org/data/lookups/CAMEO.country.txt`), while the
  `*Geo_CountryCode` fields are 2-letter **FIPS 10-4** codes (`US`, `TW`,
  `KS` -- note South Korea is `KS` in FIPS 10-4, not the ISO `KR`; see
  `https://www.gdeltproject.org/data/lookups/FIPS.country.txt`). This
  connector filters on the CAMEO codes since that's what the Actor country
  fields actually contain; both code sets are in `config/mvp_scope.py`.
  Column layout (61 columns) was verified against a live export file and
  `GDELT-Event_Codebook-V2.0.pdf`, not assumed from memory.
- **RSS** (`ingestion/rss.py`) -- live, no key needed. 12 feeds kept after an
  empirical bake-off (each candidate was actually fetched with `feedparser`
  and checked for `bozo`/entry count) -- TechCrunch, CNBC Technology, CNBC
  Top News, MarketWatch Top Stories, Taipei Times, Nikkei Asia, Yahoo
  Finance, Investing.com Stock Market News, Seeking Alpha Market News, SCMP
  Business, Channel News Asia Business, and a Google News "semiconductor"
  search feed (the only sector-specific source -- flagged as best-effort
  since it's a generated search feed, not a newsroom's own). Dropped:
  Reuters Business/Technology (DNS dead, feedburner retired), Korea Herald
  Business, Focus Taiwan, Yonhap News, Korea JoongAng Daily, Korea Times
  Business, Taiwan News (all either 404'd, redirected to malformed XML, or
  returned zero entries), and MarketWatch's feedburner-era "Real-time
  Headlines" (redundant with Top Stories). The Verge and Ars Technica parsed
  fine but were excluded for topical relevance (consumer tech, not
  business/flux signal), not for technical failure. Full detail is in the
  docstring at the top of `ingestion/rss.py`.
- **NewsAPI** (`ingestion/newsapi.py`) -- key-gated, **not live in this
  environment**. Reads `NEWSAPI_KEY` from the environment or `.env`
  (see `.env.example`); if absent, logs one warning and returns immediately
  with zero rows and zero network calls. This is the exercised path here.

## Verification performed (2026-07-18)

Ran `python -m ingestion.run_ingest` twice back-to-back:

| run | gdelt fetched/inserted/deduped | rss fetched/inserted/deduped | newsapi |
|---|---|---|---|
| 1st | 104 / 27 / 77 | 414 / 407 / 7 | 0 / 0 / 0 (skipped, no key) |
| 2nd (immediately after) | 104 / 0 / 104 | 414 / 1 / 413 | 0 / 0 / 0 (skipped, no key) |

GDELT's 2nd run inserted zero new rows because the same 15-minute export was
still the latest (GDELT hadn't published a new one yet) -- full dedup, as
expected. RSS inserted exactly 1 new row on the 2nd run: one article was
genuinely published in the ~15 seconds between runs, everything else deduped
correctly. `data/events.db` held 435 total rows after both runs (27 gdelt +
408 rss). The nontrivial dedup counts even on the *first* run (77 for GDELT,
7 for RSS) are real, not a bug: GDELT commonly emits multiple event tuples
against the same `SOURCEURL`, and some articles are aggregated by more than
one of the 12 RSS feeds -- `INSERT OR IGNORE` on the same `id` collapses
those correctly.

## Known caveats / volume

- GDELT's 15-minute export is global; the MVP country filter (`Actor1`/`Actor2`
  in {US, Taiwan, South Korea}) cut ~580 raw rows down to ~104 in one run --
  volume will vary a lot by news cycle.
- RSS feeds are polled in full on every run (no `If-Modified-Since`/ETag
  caching yet). This is no longer hypothetical now that `ingestion/scheduler.py`
  exists: at its default 900s interval, all 12 feeds get fetched in full
  every 15 minutes, indefinitely, for as long as the scheduler runs. Per-feed
  conditional GET and configurable per-feed intervals remain unimplemented --
  a real future pass, not a resolved concern.
- NewsAPI's free tier is rate- and volume-limited (100 req/day, articles
  capped at ~1 month lookback) and untested here since no key is available;
  the single OR-joined query keeps it to one request per run rather than one
  per ticker.
- Google's News search RSS is not an official/documented API -- treat it as
  best-effort and be ready to drop it if it starts blocking or rate-limiting.

## Country codes used for GDELT filtering (source of truth)

FIPS 10-4 (from `https://www.gdeltproject.org/data/lookups/FIPS.country.txt`):
`US` = United States, `TW` = Taiwan, `KS` = South Korea.

CAMEO (from `https://www.gdeltproject.org/data/lookups/CAMEO.country.txt`,
actually used for filtering `Actor1CountryCode`/`Actor2CountryCode`):
`USA` = United States, `TWN` = Taiwan, `KOR` = South Korea.
