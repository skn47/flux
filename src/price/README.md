# Price ingestion (Phase F, price half)

Pure price ingestion: pulls daily OHLCV history from Yahoo Finance (via
`yfinance`) for the 5 tracked tickers and stores it in a clean, deduplicated
SQLite table. Does no flux/event logic, no feature engineering, no LSTM
windowing, no joining to `flux_score` -- that is separate, later work. This
package's only job is to ingest faithfully and never fabricate or silently
backfill data, mirroring `ingestion/`'s (Phase A) discipline.

## Tracked tickers

`NVDA, TSM, AMD, ASML, INTC` -- read from `config.mvp_scope.TRACKED_TICKERS`,
never hardcoded a second time, so the two packages can't silently drift apart.

## Schema

`price.schema.PriceBar`: `ticker, date, open, high, low, close, adj_close,
volume, source, ingested_at`.

Unlike `ingestion.schema.RawEvent` (which needs a synthetic sha256 dedup id
because news records have no natural key), a daily bar already has a natural
stable key: `(ticker, date)`. No hash is manufactured -- that pair is used
directly as the SQLite primary key.

`PriceBar.validate()` checks: date is well-formed `YYYY-MM-DD`, no field is
`None`/NaN, no field is negative, and `low <= {open, close} <= high`. Rows
that fail validation are dropped and logged (`skipped_invalid_rows`), never
patched into looking valid.

**`close` vs `adj_close`:** both are stored, deliberately not collapsed into
one column. `close` is the raw/unadjusted price as printed on the tape;
`adj_close` is yfinance's split/dividend-adjusted close. A downstream feature
pipeline building return series should almost always use `adj_close` (raw
`close` has artificial jumps at every split -- e.g. NVDA's 10:1 split in 2024
-- that are not real price moves). This ingestion stage takes no position on
which one a future feature pipeline should use; that's out of scope here.

**Date discipline (read `price/schema.py`'s module docstring for the full
writeup):** `date` is a trading-day calendar date (`YYYY-MM-DD`), not an
instant in time, and carries no time-of-day. yfinance's daily index is a
midnight timestamp localized to the exchange's local timezone. We take
`.date()` directly off that localized timestamp with **no conversion to
UTC** -- converting first would shift the date backward by one calendar day
for roughly half the year (whenever the local UTC offset is negative),
silently mislabeling every bar. This was checked, not assumed: all 5 tracked
tickers' yfinance history is indexed in `America/New_York`, including the
ADRs TSM and ASML (verified empirically 2026-07-21; they are NOT indexed in
their home-exchange timezones, Taiwan Stock Exchange / Euronext Amsterdam --
yfinance serves them as their US-listed ADR sessions).

## Storage

SQLite, same file as every other phase in this project: `data/events.db`,
new table `daily_prices`, primary key `(ticker, date)`. **Chose to reuse
`events.db` rather than a separate `data/prices.db`** because:
- every other phase (A/B/D/E) already shares that one file; a new file would
  be the first exception to an established pattern, not a continuation of
  one.
- Phase F's entire purpose is joining `flux_score(stock, t)` against price
  history for matching `(ticker, date)` pairs -- keeping both in one file
  means that join is a single `sqlite3` connection, no `ATTACH DATABASE`.
- the new table is fully independent (own primary key, no foreign keys into
  `raw_events`/`classified_events`), so there's no schema entanglement risk
  from sharing the file.

Inserts use `INSERT OR IGNORE` keyed on `(ticker, date)`, so re-running
ingestion is idempotent by construction -- confirmed by running it twice
back-to-back (see Verification below). `price/storage.py`'s `PriceStore`
also has `date_range(ticker)` and `recent()` for sanity-checking.

## Running it

```
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m price.run_ingest
```

Pulls `period="max"` (yfinance's full available history, not just the last
couple of years) daily OHLCV for all 5 tickers, catching exceptions
per-ticker (one broken/rate-limited ticker can never take down the others),
and prints a fetched/inserted/deduped/date-range/error summary per ticker.
Exit code is non-zero only if every single ticker failed outright.

## Real run performed (2026-07-21)

Command actually executed against the live network:

```
./.venv/bin/python -m price.run_ingest
```

Script's own printed summary (first run, empty table):

| ticker | fetched | inserted | deduped | min_date | max_date |
|---|---|---|---|---|---|
| NVDA | 6915 | 6915 | 0 | 1999-01-22 | 2026-07-21 |
| TSM | 7238 | 7238 | 0 | 1997-10-09 | 2026-07-21 |
| AMD | 11680 | 11680 | 0 | 1980-03-17 | 2026-07-21 |
| ASML | 7889 | 7889 | 0 | 1995-03-15 | 2026-07-21 |
| INTC | 11680 | 11680 | 0 | 1980-03-17 | 2026-07-21 |

Re-ran immediately after, to check idempotency: all 5 tickers showed
`fetched == deduped` and `inserted == 0` (e.g. NVDA: 6915 fetched / 0
inserted / 6915 deduped) -- no duplicate rows were created.

**Independently re-verified by querying `data/events.db` directly** (not
trusting the script's own printed summary), using
`GROUP BY ticker` / `COUNT(DISTINCT date)` / a duplicate-key check / a
NULL-check on all numeric columns:

| ticker | row count | min date | max date | duplicate (ticker,date) rows | NULL/NaN rows |
|---|---|---|---|---|---|
| NVDA | 6915 | 1999-01-22 | 2026-07-21 | 0 | 0 |
| TSM | 7238 | 1997-10-09 | 2026-07-21 | 0 | 0 |
| AMD | 11680 | 1980-03-17 | 2026-07-21 | 0 | 0 |
| ASML | 7889 | 1995-03-15 | 2026-07-21 | 0 | 0 |
| INTC | 11680 | 1980-03-17 | 2026-07-21 | 0 | 0 |

Total: **45,402 rows** in `daily_prices`, matching the sum of the per-ticker
counts above exactly. `COUNT(DISTINCT date)` equaled `COUNT(*)` for every
ticker (no duplicate trading days), and the direct DB query's numbers match
the script's self-reported numbers exactly -- no discrepancy found.

No ticker failed. All 5 pulls succeeded against the live yfinance/Yahoo
Finance API in this run.

## Known gaps / limitations (be honest about these, don't round up)

- **History length differs per ticker and is NOT ~2 years like the event
  data** -- it's whatever yfinance/Yahoo actually has for that ticker's US
  listing:
  - NVDA: 1999-01-22 onward (6,915 trading days)
  - TSM (ADR): 1997-10-09 onward (7,238 trading days)
  - AMD: 1980-03-17 onward (11,680 trading days)
  - ASML (ADR): 1995-03-15 onward (7,889 trading days)
  - INTC: 1980-03-17 onward (11,680 trading days)

  These are Yahoo Finance's own available-history limits for each symbol,
  not a bug or a deliberate truncation on our end. TSM and ASML's histories
  start when their respective ADRs began trading on a US exchange, not when
  the underlying company itself started trading (TSMC has traded on the
  Taiwan Stock Exchange since 1994, ASML on Euronext since 1995 -- their
  pre-ADR home-exchange history is not in this dataset at all, since we
  never fetched the home-exchange tickers `2330.TW` / `ASML.AS`). If a
  future spec needs longer or home-exchange history for TSM/ASML, that is a
  different ticker symbol and a separate ingestion decision, flagged here
  rather than silently substituted.
- **No forward-fill applied by this stage, on purpose.** The task's default
  is forward-fill for missing data, but that applies to feature engineering
  over a fixed calendar grid, not to raw ingestion. This stage stores
  exactly the trading days yfinance returns for each ticker and does not
  invent rows for non-trading days or for gaps. Aligning the 5 tickers onto
  a single shared trading-day calendar (they may not have identical trading
  calendars, e.g. US market holidays vs. any residual differences) and
  deciding how to handle any genuine within-calendar gaps is deferred to
  whatever feature-engineering/windowing stage consumes this table next --
  flagging this explicitly rather than making that call here.
- **Adjustment caveat:** `adj_close` is yfinance's own split/dividend
  adjustment, computed as of the time this was fetched (2026-07-21). Adjusted
  closes for a given historical date can change on a LATER fetch if a new
  corporate action (e.g. a future stock split) occurs after that date --
  this is normal for split-adjusted series and is why `close` (unadjusted,
  immutable) is also stored alongside it. Spot-checked: for the most recent
  trading day in every ticker, `close == adj_close` (no adjustment pending),
  while for early-history rows `adj_close < close` (reflecting subsequent
  splits/dividends) -- e.g. NVDA's first stored bar (1999-01-22) has
  `close=0.041016`, `adj_close=0.037559`, consistent with NVDA's multiple
  splits since IPO.
- **Zero-volume days exist in the raw AMD/INTC history** (2 days for AMD, 1
  for INTC, out of tens of thousands of rows) -- stored as-is (real reported
  volume of 0), not treated as missing/invalid, since `PriceBar.validate()`
  only rejects negative volume, not zero.
- **Survivorship bias**: this dataset only reflects tickers that exist and
  trade under their current symbol today. It says nothing about delisted
  competitors, ticker changes, or corporate restructurings (e.g. spinoffs)
  in the tracked names' history -- a general yfinance/Yahoo Finance
  limitation, not something this ingestion stage attempts to correct for.
- **No intraday data** was pulled or is supported by this package, by
  design (task spec: daily only).
- **No retry/backoff logic** for yfinance rate limits yet -- this run hit no
  rate limiting, but a future scheduled/automated run should add one before
  relying on this unattended (mirrors `ingestion/README.md`'s caveat about
  RSS lacking conditional-GET caching).

## Files

- `price/schema.py` -- `PriceBar` dataclass + validation.
- `price/storage.py` -- `PriceStore`, SQLite table `daily_prices` in
  `data/events.db`, idempotent `INSERT OR IGNORE` writes.
- `price/fetch.py` -- yfinance connector, per-ticker error isolation, no
  fabricated/interpolated bars.
- `price/run_ingest.py` -- CLI entry point (`python -m price.run_ingest`).
