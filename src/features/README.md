# Feature building (Phase F, feature-table half)

Joins `daily_prices` (Phase F, price half) against `flux_scores_daily`
(Phase F, flux half) into a single, per-ticker, per-trading-day, **unscaled**
feature table for the LSTM, per the design decisions in
`progress/phase_f_lstm_decisions.md`. Does no scaling, no windowing into actual
tensors, no model code, no training loop -- those are later steps (the next
one being LSTM training, owned by the Deep Learning Architect agent).

## Feature vector

Per `(ticker, date)`, six raw (unscaled) numeric columns:

| column | definition | why |
|---|---|---|
| `adj_close` | split/dividend-adjusted close (price level) | `price/README.md` explains why `adj_close`, not raw `close`: raw close has artificial jumps at every split (e.g. NVDA's 10:1 split in 2024) that are not real price moves. |
| `daily_return` | `pct_change(adj_close, 1)` vs. the prior **trading** day | single-day simple return; standard, cheap, causal by construction (pandas `.pct_change()` only looks backward). |
| `sma_short` | 10-trading-day trailing simple moving average of `adj_close` | fast trend read; classic short leg of a 10/50 "golden/death cross" pair. |
| `sma_long` | 50-trading-day trailing simple moving average of `adj_close` | slow trend read; long leg of the same pair. |
| `volatility_10` | 10-trading-day trailing rolling std of `daily_return` | realized volatility, a standard measure of recent turbulence. |
| `flux_score` | `flux_scores_daily.flux_score` for this ticker/date | the single external signal per `progress/phase_f_lstm_decisions.md` decision #2 -- supersedes a large macro/fundamental dump. |

**Deliberately excluded, to keep the vector lean** (decision #2: "price/technical
features + a single external signal ... not a large feature dump"):
- Raw `open`/`high`/`low`/`volume` as separate columns. `adj_close` and
  `volatility_10` are both literally OHLCV-derived (from `close` history);
  adding four more raw series would reintroduce the kind of
  correlated/redundant feature bloat that `progress/phase_f_lstm_decisions.md`
  cites as the reason Paper 2's macro dump hurt accuracy.
- `direction`, `direction_coverage`, `n_clusters` from `flux_scores_daily`
  (diagnostic/debugging columns from the flux engine itself, not part of
  the model's feature vector).
- Any centered or non-causal indicator (e.g. a centered moving average) --
  every column above uses pandas `.rolling()`/`.pct_change()`, both
  trailing-only by construction, so no column at row `t` can read row `t+1`
  or later.

All six columns are stored **raw/unscaled**. Scaling (e.g. Min-Max or
Z-score) is deliberately deferred to the LSTM training step, per
`progress/phase_f_lstm_decisions.md` decision #3: scalers must be fit on the
train partition only and applied to val/test with those fitted parameters --
that can only happen correctly once the actual windowed train tensors exist,
which is out of scope here. Fitting a scaler on this raw table (or on the
full dataset) here would be exactly the leakage bug the decisions doc calls
out in Paper 1's code.

## Join logic: flux_score onto the trading-day calendar

`daily_prices` is the **authoritative trading-day calendar** -- one feature
row per real trading day, never a synthesized/interpolated row for a
non-trading day. `flux_scores_daily` has one row per **calendar** day
(GDELT events happen every day, including weekends/holidays), so for each
trading day we look up `flux_scores_daily` at that exact calendar date.

Verified against the live DB (2026-07-22, see "Verification" below): **every
one of the 502 trading days per ticker inside `flux_scores_daily`'s calendar
coverage (2024-07-19 .. 2026-07-21) has an exact date match** -- 0 missing,
0 forward-fills needed, for all 5 tickers. The task spec anticipated a
possible forward-fill fallback ("if a trading day has no exact-date match,
forward-fill from the most recent prior calendar date's flux_score"); the
code (`features/build.py::_join_flux_score`) implements that fallback
defensively (kept for correctness against a future GDELT gap on a single
calendar day) but **it is not exercised by the current, real dataset** --
confirmed by an independent DB query (`flux_score_source_date != date`
count = 0, see below).

**The real bottleneck is coverage, not join mechanics**: `flux_scores_daily`
only goes back to 2024-07-19, far short of every tracked ticker's price
history (NVDA back to 1999, AMD/INTC back to 1980, etc. -- see
`price/README.md`). Since `flux_score` is a required feature per the
decisions doc (every timestep needs it, not an optional/nullable column),
**the feature table is bottlenecked to 2024-07-19 .. 2026-07-21 (502 trading
days) for every ticker**, regardless of how much earlier price history
exists. This is the single biggest caveat of this dataset -- flagged
explicitly, not silently absorbed. Rows before 2024-07-19 are never
fabricated a flux_score; they're simply not part of the output table (they
are, however, still used internally to warm up the technical indicators --
see below).

## No leakage

Every numeric feature at row `(ticker, t)` only uses information available
as of the close of trading day `t`:
- `daily_return`, `sma_short`, `sma_long`, `volatility_10` are all computed
  with pandas `.pct_change()` / `.rolling(window).mean()` /
  `.rolling(window).std()` over the FULL per-ticker price history, sorted
  ascending -- all three are trailing-only (never centered), so a value at
  row `t` never reads row `t+1` or later.
- `flux_score` is joined at the exact calendar date `t` (or, if that
  fallback ever fires, from a **prior** calendar date -- never a later one).
  `FeatureRow.validate()` enforces this as a hard invariant:
  `flux_score_source_date <= date`, raising `InvalidFeatureRowError`
  otherwise. Every one of the 2,510 stored rows passed this check.
- Technical indicators are computed over each ticker's **entire** available
  price history (not just the 2024-07-19-onward window) specifically so that
  the 10-day/50-day rolling warmup lands in the pre-2024 past, not in the
  output table. Verified: 0 rows were dropped for insufficient warmup
  (`warmup_nan_rows_dropped = 0` for all 5 tickers) -- every ticker has far
  more than 50 trading days of price history before 2024-07-19.
- No feature here is a function of any future target. This step doesn't
  compute a label at all (that's the LSTM training step's job); the caveat
  is recorded here so the label-construction step doesn't accidentally
  define a label at day `t` using `sma_short`/`sma_long` computed with a
  window that includes day `t+1` -- it doesn't, by construction, but this is
  worth restating since it's the exact failure mode the task guards against.

## Chronological train/val/test split

Ratios are a documented implementation choice (decision #4 leaves exact
ratios open): **70% train / 15% val / 15% test, by row count**, applied
per-ticker over the 502-row feature table, contiguous and non-overlapping.
On the real data every ticker has an **identical** 502-trading-day calendar
in the flux window (verified independently, see below), so the resulting
date boundaries are identical across all 5 tickers:

| split | start | end | n_rows |
|---|---|---|---|
| train | 2024-07-19 | 2025-12-10 | 351 |
| val | 2025-12-11 | 2026-03-31 | 75 |
| test | 2026-04-01 | 2026-07-21 | 76 |

70/15/15 (rather than Paper 2's cited 60/20/20) was chosen because the
flux-score join bottleneck already caps this dataset at 502 rows/ticker --
a smaller train fraction would leave a scaler/model with very little data to
fit on, while 75-76 rows is still a workable val/test size for early
stopping and final evaluation.

**Why a window's features may cross a split boundary but its label never
can**: a supervised window with label date `L` uses feature days
`[L - 60 .. L - 1]` (60 trading days, all strictly before `L`) and a label
from `L` or later. Because feature days are always chronologically before
`L` by construction, if `L` falls in `train` (i.e. `L` < val start), every
feature day of that window is automatically also before val start -- a
train window can never see val/test-period data. A `val` window's feature
days, however, may legitimately reach back into `train`-period dates (e.g.
the very first usable val-labeled window needs the 60 trading days
immediately preceding it, some of which are dated in `train`) -- **this is
not leakage**, it's real, already-realized historical context, and is the
normal/expected behavior of any sliding window evaluated near a
chronological split boundary. It is never allowed to run the other
direction (a `train` window peeking into `val`/`test`).

**60-trading-day lookback vs. row availability**: the first 60 rows of the
overall per-ticker series (2024-07-19 through the 60th trading day, 2024-10-11)
can only ever serve as feature context, never as a window's own label --
there isn't 60 days of prior history *inside this table* before them. This
means `train`'s first 60 rows produce no usable label, but every date from
row 61 onward (`train`, `val`, and `test` alike) does. Stored explicitly in
`feature_split_bounds.first_usable_label_date` /
`.n_usable_labels` per (ticker, split) -- e.g. for NVDA: `train` has 291
usable label dates (351 - 60), `val` has all 75, `test` has all 76, summing
to 442 = 502 - 60 total usable supervised windows per ticker. This does
**not** need to be (and per the task, isn't) materialized as actual windowed
`(60, features)` tensors here -- that's the LSTM training step's job,
reproducible from `feature_split_bounds` and the `split` column on
`features_daily` alone.

## Storage

SQLite, same file as every other phase in this project: `data/events.db`
(not a separate file, not Parquet). Chose this, matching `price/README.md`'s
reasoning for the same call:
- every other phase (A/B/D/E, plus the price half of this phase) already
  shares `events.db`; a new file/format would be the first exception, not a
  continuation of the established pattern.
- this step's entire purpose is a join between two tables (`daily_prices`,
  `flux_scores_daily`) that already live in `events.db` -- staying in the
  same file means the join is one `sqlite3` connection with plain SQL, no
  `ATTACH DATABASE`, no Parquet-then-pandas-merge indirection.
- the data volume is small (2,510 rows total across 5 tickers) -- nowhere
  near the scale where SQLite's per-query overhead or lack of columnar
  compression would matter; Parquet's main advantages (columnar scan
  performance, partitioning) aren't relevant at this size.

Two new tables:
- **`features_daily`**: one row per `(ticker, date)` -- `ticker, date,
  adj_close, daily_return, sma_short, sma_long, volatility_10, flux_score,
  flux_score_source_date, split, computed_at`. Primary key `(ticker,
  date)`. Writes use `INSERT OR REPLACE` (not `INSERT OR IGNORE`) --
  unlike `daily_prices`' immutable raw ingestion, these are *derived*
  values; if the indicator formulas or split ratios ever change, re-running
  this build should overwrite stale rows, matching `flux_scores_daily`'s
  own storage convention (`flux_engine/timeseries.py::store_daily_series`),
  not `daily_prices`' append-only one.
- **`feature_split_bounds`**: one row per `(ticker, split)` -- the actual
  date-range boundaries, row counts, and usable-label counts used, so the
  3-way split is reproducible from the DB alone without re-deriving it from
  row order.

## Running it

```
./.venv/bin/python -m features.run_build
```

Builds and upserts feature rows + split bounds for all 5
`config.mvp_scope.TRACKED_TICKERS`, isolating per-ticker failures (mirrors
`price/run_ingest.py`) and printing a summary table.

## Real run performed (2026-07-22)

Command actually executed against the live `data/events.db`:

```
./.venv/bin/python -m features.run_build
```

Script's own printed summary:

| ticker | rows | min_date | max_date | train | val | test | ffill | unmatched | warmup_nan |
|---|---|---|---|---|---|---|---|---|---|
| NVDA | 502 | 2024-07-19 | 2026-07-21 | 351 | 75 | 76 | 0 | 0 | 0 |
| TSM | 502 | 2024-07-19 | 2026-07-21 | 351 | 75 | 76 | 0 | 0 | 0 |
| AMD | 502 | 2024-07-19 | 2026-07-21 | 351 | 75 | 76 | 0 | 0 | 0 |
| ASML | 502 | 2024-07-19 | 2026-07-21 | 351 | 75 | 76 | 0 | 0 | 0 |
| INTC | 502 | 2024-07-19 | 2026-07-21 | 351 | 75 | 76 | 0 | 0 | 0 |

Re-ran immediately after to check idempotency: total row count in
`features_daily` stayed at 2,510 (0 net new/changed rows on the second run
against unchanged upstream data).

**Independently re-verified by querying `data/events.db` directly**
(not the script's self-report):

| check | result |
|---|---|
| Row count per ticker | AMD 502, ASML 502, INTC 502, NVDA 502, TSM 502 -- all 2024-07-19 to 2026-07-21 |
| Total rows in `features_daily` | **2,510** (= 5 x 502, matches exactly) |
| Duplicate `(ticker, date)` rows | **0** |
| NULL rows across all 6 required numeric columns | **0** |
| Leakage guard: rows where `flux_score_source_date > date` | **0** |
| Rows where `flux_score_source_date != date` (i.e. forward-filled) | **0** -- confirms the exact-date join applies 100% of the time on this dataset |
| `features_daily.flux_score` vs. `flux_scores_daily.flux_score` (joined re-check, all 2,510 rows) | **0 mismatches** |
| `feature_split_bounds` boundaries | identical across all 5 tickers: train 2024-07-19..2025-12-10 (351 rows), val 2025-12-11..2026-03-31 (75 rows), test 2026-04-01..2026-07-21 (76 rows) |
| `n_usable_labels` sum (train 291 + val 75 + test 76) vs. `502 - 60` | both equal **442** |
| Hand-recomputed `sma_short`/`sma_long`/`daily_return` for NVDA on 2024-07-19 directly from `daily_prices` (independent pandas calc, not reusing `features/indicators.py`) | matched the stored values exactly: `sma_short=126.086180...`, `sma_long=115.959261...`, `daily_return=-0.026096...` |

No ticker failed. All 5 builds succeeded against the live `data/events.db`
in this run.

## Known gaps / limitations (be honest about these, don't round up)

- **Effective usable history is only ~2 years (502 trading days), not each
  ticker's full price history**, because `flux_scores_daily` (a required
  feature per the decisions doc) only exists from 2024-07-19 onward. This
  applies equally to all 5 tickers even though NVDA/TSM/AMD/ASML/INTC have
  wildly different total price history lengths (see `price/README.md`) --
  once flux_score is required, that longer history is only usable as
  hidden warmup for the technical indicators, never as additional
  supervised rows. If a future spec allows a price-only (no flux_score)
  baseline model, that would use a much longer, ticker-specific history and
  is a different build, not covered here.
- **Only 442 usable 60-day supervised windows per ticker** (502 - 60), split
  291/75/76 across train/val/test. This is a small dataset for an LSTM by
  general deep-learning standards; flagged here as a downstream training
  consideration (e.g. risk of overfitting, need for regularization), not
  something this build step can fix by fabricating more data.
- **`daily_return` has some large single-day swings that are real, not data
  errors** -- spot-checked the extremes: INTC -26.06% on 2024-08-02 (Intel's
  Q2 2024 earnings miss/guidance cut, a real, widely reported crash) and AMD
  +23.8% on 2025-04-09 (a market-wide rally day). Not clipped, winsorized,
  or treated as outliers here -- that's a modeling decision for the training
  step if it turns out to matter, not a data-quality issue to silently fix.
- **`flux_score` range in the stored table is [0.1417, 0.9866]**, i.e. it
  never reaches the formula's theoretical [0, 1] extremes in this window --
  consistent with `flux_engine`'s noisy-OR construction (see
  `flux_engine/README.md`) and not itself a bug, just noted as a real,
  checked characteristic of the data rather than assumed.
- **Weekday-only trading calendar**: `daily_prices` (and therefore this
  feature table) has no rows for weekends or US market holidays, by design
  -- `flux_scores_daily` has a row for every calendar day, and this join
  intentionally uses `daily_prices` as the authoritative calendar rather
  than backfilling synthetic non-trading-day rows.
- **Data as of 2026-07-21** (the latest date in both source tables at build
  time) -- this feature table will need to be rebuilt (idempotently; safe to
  re-run) whenever `daily_prices`/`flux_scores_daily` are refreshed with
  newer dates.
- **No scaling applied, by design** -- restated here one more time because
  it's the easiest rule for a downstream step to accidentally violate: fit
  any scaler ONLY on rows where `split = 'train'`, then `.transform()` (not
  `.fit_transform()`) the `val`/`test` rows with those fitted parameters.

## Files

- `features/schema.py` -- `FeatureRow` dataclass (with the leakage-guard
  `validate()`) and `SplitBounds` dataclass.
- `features/indicators.py` -- causal technical-indicator computation
  (`daily_return`, `sma_short`, `sma_long`, `volatility_10`), pure pandas,
  trailing-only.
- `features/build.py` -- per-ticker orchestration: load full price history,
  compute indicators, join `flux_score` (exact-date with a defensive
  forward-fill fallback), restrict to the flux-covered window, assign the
  chronological 3-way split, validate every row.
- `features/storage.py` -- `FeatureStore`, SQLite tables `features_daily`
  and `feature_split_bounds` in `data/events.db`, idempotent
  `INSERT OR REPLACE` writes.
- `features/run_build.py` -- CLI entry point
  (`python -m features.run_build`).
