# LSTM price-reaction model (Phase F, final step; reworked under R1, walk-forward-validated under R5)

## R5 addendum (2026-07-23) — walk-forward validation — READ THIS FIRST for the current headline result

R1-R4 (below) answered "does the LSTM have cross-sectional skill" on exactly
ONE 76-trading-day test window (2026-04-02..2026-07-22) -- one correlated
sector-wide-rally episode. R5 (see
`/home/h/.claude/plans/yes-kick-off-phase-stateful-trinket.md`, "R5"
section) re-runs training + backtesting across **4 expanding-window folds**
carved from the full 2024-07-19..2026-07-22 history (503 trading days/ticker,
443 usable label days after `LOOKBACK=60`), retraining EVERY model fresh
from scratch per fold (never warm-started -- see `backtest/walk_forward.py`'s
module docstring for why that matters for expanding-window folds
specifically). This is a **directional robustness check across 4
sub-periods, not a large-N study** -- restated here and in every place a
number from this section is quoted.

### Code changes (this section only; R1-R4's own logic untouched)

- `lstm/dataset.py::build_ticker_dataset`/`build_pooled_dataset` gained an
  optional `split_boundaries` parameter (default `None`). `None` reproduces
  the exact pre-R5 code path (reads the stored `features_daily.split`
  column) -- **regression-checked, not just claimed**: re-ran
  `./.venv/bin/python -m lstm.run_train` (single-window path) after adding
  the parameter and diffed `lstm/models/run_summary.json` against a copy
  saved before the change -- **byte-identical** (`diff` returned no output).
  Same regression check performed on
  `./.venv/bin/python -m backtest.run_cross_sectional_backtest` against
  `backtest/cross_sectional_metrics.csv`/`.json` (also exercises the
  `_run_cross_sectional_engine` -> `run_cross_sectional_engine` relocation
  into `backtest/cross_sectional.py`, done so `backtest/walk_forward.py`
  could reuse it without forking) -- **also byte-identical**.
- New `lstm.dataset.split_for_date(date, split_boundaries)`: classifies one
  raw date string against a fold's boundaries; a row whose date falls after
  a fold's own test end gets no split at all (not built into any window) --
  this is the mechanism that truncates each fold's dataset to only what
  that fold's own history cutoff could have seen.
- New `lstm/walk_forward_bounds.py`: `walk_forward_bounds` DB table (keyed
  `(fold_id, ticker, split)` -> `(start_date, end_date)`, 96 rows written
  this run = 4 folds x 8 tickers x 3 splits) so the leakage check has a
  persisted, re-queryable source of truth per fold, not an in-memory
  assumption; `verify_fold_boundaries()` (asserts each fold's
  train<val<test ordering AND no cross-fold test overlap);
  `lstm.dataset.verify_no_leakage(..., fold_id=k)` (fold-scoped variant of
  the existing check, reads `walk_forward_bounds` instead of
  `feature_split_bounds`).
- `backtest/predictions.py::generate_test_predictions` gained
  `checkpoint_path`/`split_boundaries` parameters (both `None`-default,
  backward compatible) so `backtest/walk_forward.py` reuses this exact
  function per fold instead of forking prediction-generation logic.
- New `backtest/walk_forward.py`: orchestrates the whole fold loop
  (boundary computation from the live DB, per-fold training x2 variants,
  per-fold cross-sectional backtest, aggregate report) -- see
  `backtest/README.md`'s R5 addendum for the full results table, hit-rates,
  and verdict.

### Fold boundaries (computed from the live DB, not hardcoded)

| fold | train (label days) | val (label days) | test (label days) |
|---|---|---|---|
| 1 | 2024-07-19..2025-08-12 (207) | 2025-08-13..2025-10-02 (36) | 2025-10-03..2025-12-12 (50) |
| 2 | 2024-07-19..2025-10-10 (249) | 2025-10-13..2025-12-12 (44) | 2025-12-15..2026-02-26 (50) |
| 3 | 2024-07-19..2025-12-11 (292) | 2025-12-12..2026-02-26 (51) | 2026-02-27..2026-05-08 (50) |
| 4 | 2024-07-19..2026-02-12 (334) | 2026-02-13..2026-05-08 (59) | 2026-05-11..2026-07-22 (50) |

Each fold's TRAIN always starts at the true global first date (2024-07-19,
matching `feature_split_bounds`' own convention of using the first raw row,
not the first *usable-label* date) -- this is what makes the scheme
"expanding": fold 4's train pool is a strict superset of fold 1's. The 4
test slices partition the LAST 200 of the 443 usable label days with **zero
gap, zero overlap** (fold k+1's test start is the very next usable label
day after fold k's test end for k=1,2; a normal calendar gap for k=3 since
2026-05-08 -> 2026-05-11 spans a weekend, still zero *usable-label-day*
gap). VAL is the trailing ~15% of each fold's own expanding train-to-date
pool (mirrors the existing single-window 70/15/15 convention).

**Independently spot-checked by hand** (not just trusted from the script's
own printed output): re-queried `features_daily` directly for fold 2/TSM,
recomputing "usable label days in each date range" from scratch (a date is
usable iff its position in the full ascending date list is >= 60) --
**train=249, val=44, test=50, matching `walk_forward_bounds` exactly**; also
confirmed fold 2's test_start (2025-12-15) is strictly after fold 1's
test_end (2025-12-12), and fold 3's test_start (2026-02-27) is strictly
after fold 2's test_end (2026-02-26).

### Verification results (zero violations, every check)

- `verify_fold_boundaries(folds)`: `{'n_folds_checked': 4, 'status': 'no
  fold-boundary violations found'}`.
- `verify_no_leakage(by_split, fold_id=k)` for every one of the 8 (4 folds x
  2 variants) fold/variant combinations trained -- all returned `'status':
  'no leakage found'`, with window counts that sum correctly to each fold's
  own boundaries x 8 tickers (e.g. fold 3: `{'train': 2336, 'val': 408,
  'test': 400}` = `(292+51+50) x 8`).
- `verify_dollar_neutral` -- **13 calls/fold** (2 LSTM variants + 1
  sma_momentum + 10 random-rank seeds), **52/52 passed** across all 4 folds,
  zero violations.
- Hand-verified one full day's weight vector (fold 1, baseline,
  2025-12-12): `INTC +0.21875, QCOM +0.15625, TSM +0.09375, AMAT +0.03125,
  ASML -0.03125, NVDA -0.09375, MU -0.15625, AMD -0.21875` --
  `sum(weight)=0.0`, `sum(|weight|)=1.0`, matching
  `raw_score_i=(8+1)/2-rank_i` by hand.

### Per-fold training summary (fresh model every cell, never warm-started)

| fold | variant | train/val/test windows | best_epoch | epochs trained |
|---|---|---|---|---|
| 1 | baseline | 1656/288/400 | 1 | 16 |
| 1 | flux | 1656/288/400 | 3 | 18 |
| 2 | baseline | 1992/352/400 | 2 | 17 |
| 2 | flux | 1992/352/400 | 3 | 18 |
| 3 | baseline | 2336/408/400 | 6 | 21 |
| 3 | flux | 2336/408/400 | 9 | 24 |
| 4 | baseline | 2672/472/400 | 1 | 16 |
| 4 | flux | 2672/472/400 | 2 | 17 |

Fold 1 and fold 4 both have `best_epoch=1`/`best_epoch=2` -- val loss
stopped improving almost immediately, then ran the full
`early_stop_patience=15` before triggering (16-17 total epochs). This is
plausible, not obviously a bug: fold 1 has the smallest train pool (207
label days/ticker) and fold 4's val slice is the least similar in market
regime to its own tiny incremental train extension -- neither is a change
in `lstm/train.py`'s logic (unchanged by R5, reused as-is; every one of
these 8 models used the exact same `TrainConfig()` defaults, no per-fold
retuning, per the task's explicit "comparability, not squeezing extra
performance" instruction).

See `backtest/README.md`'s R5 addendum for the full per-fold cross-sectional
backtest table, aggregate hit-rates, and the explicit verdict.

---

Trains and evaluates the next-day return-prediction LSTM in two feature
variants -- **baseline** (price/technical only) and **flux-augmented** (+
`flux_score`) -- on `features_daily` (`data/events.db`, built by
`features/`, see `features/README.md`).

**R1 rework notice (2026-07-23):** this package originally predicted the
next trading day's raw `adj_close` **price level**. That design was
identified as the root cause of Phase G's backtest failure (both LSTM
variants lost ~34% net of costs against a +71% buy-and-hold, because all 5
tracked tickers rallied past the train-fitted scaler's price range and both
models stayed net-short through the rally -- see the old numbers preserved
in `backtest/README.md`'s Part 2). R1 (see
`/home/h/.claude/plans/yes-kick-off-phase-stateful-trinket.md`, "Status
update (2026-07-23)") reframes both the **target** and the **input
features** around `daily_return` and price-relative ratios instead of an
absolute price level. This README documents the reworked design and reports
the R1 gate check's result honestly (see "Gate check result" below) -- the
old price-level numbers are NOT deleted from `backtest/README.md`, which
still documents the original failure for the record.

## Framework choice

**PyTorch** (CPU-only, `torch==2.13.0+cpu`, already installed and pinned in
`requirements.txt` from Phase C's `nlp_classifier/`). Unchanged by R1.

## Target / feature framing (rewritten under R1)

**Target: next trading day's `daily_return`** (a simple, single-day return),
not a price level. **Input features are also reframed** away from raw
`adj_close`:

    BASELINE_FEATURE_COLS = [daily_return, price_to_sma_long, sma_ratio,
                              volatility_10]                        (4 cols)
    FLUX_FEATURE_COLS    = BASELINE_FEATURE_COLS + [flux_score]   (5 cols)
    TARGET_COL = daily_return   (always index 0 in both variants)

`price_to_sma_long = adj_close / sma_long` and `sma_ratio = sma_short /
sma_long` are new columns, computed **in-memory** in
`lstm/dataset.py::load_ticker_frame` from `features_daily`'s already-stored
`adj_close`/`sma_short`/`sma_long` -- the `features_daily` DB schema and
`features/build.py` are untouched (per `features/README.md`, that package
defers scaling/derived-ratio work to the consumer).

**Why the old design failed, precisely**: an absolute price level, used as
both a per-timestep input and the regression target, scaled by a
**train-only-fit range**, is *structurally guaranteed* to leave that fixed
range once a stock makes a new all-time high -- which any sustained rally
eventually causes. This is not a bug in the train-only-fit discipline (that
discipline is correct and non-negotiable, see "Scaling" below); it's a
target/feature framing choice that made the failure mode inevitable given a
long enough rally.

**R1's hypothesis** (tested, not assumed, by the gate check below): a daily
return and a price-to-moving-average ratio both stay in a comparatively
narrow, roughly regime-independent band even during a strong trend, because
they measure relative/local behavior (how far price moved today; how far
price has stretched above its own trend) rather than an absolute level that
grows without bound. This is explicitly **not a guarantee** -- an
unprecedented, still-accelerating trend can still push `price_to_sma_long`
outside its train range, just (per the hypothesis) far less severely than
raw price level did, since the ratio's entire point is to net out the trend
itself, not to be immune to how far price can stretch from it.

The cost of the OLD framing, for reference (previously in this section):
predicting a price level directly reproduces both source papers' stated
target and lets MAPE be directly compared to their reported figures, at the
cost of the extrapolation failure above. R1 gives up that direct MAPE
comparability to the papers in exchange for addressing the actual root
cause.

## Windowing rule (leakage-safe by construction, UNCHANGED by R1)

A window with label date `L` uses feature rows `[L-60 .. L-1]` (60 trading
days, strictly before `L`) and label = `daily_return` at `L` (was `adj_close`
before R1). The window's split is whatever `features_daily.split` says for
row `L` **only** -- never derived from a majority vote over the window's
feature rows. R1 did not touch this logic at all (per its own scope) -- see
`lstm/dataset.py`'s module docstring and `verify_no_leakage()`, and the
"Verification performed" section below confirming the exact same window
counts as before.

## Feature variants (rewritten under R1)

| variant | columns | n_features |
|---|---|---|
| `baseline` | `daily_return, price_to_sma_long, sma_ratio, volatility_10` | 4 |
| `flux` | baseline + `flux_score` | 5 |

Same windows, same labels, same split assignment for both -- only the
feature vector differs, so the comparison isolates `flux_score` as the
single variable under test (unchanged design principle from before R1).

## Scaling (rewritten under R1: `StandardScaler`, not `MinMaxScaler`)

A per-**(ticker, variant)** z-score `StandardScaler` (`lstm/scaling.py`, a
new from-scratch class alongside the still-present `MinMaxScaler`), fit
**only** on that ticker's `split == "train"` rows, then applied (never
re-fit) to that ticker's full row range before windows are sliced out.
`lstm/dataset.py::build_ticker_dataset` is the only place `.fit()` is ever
called in this package, and it is always called on a train-only slice --
this structurally guarantees the "fit scalers on the train partition only"
rule, unchanged from before R1.

**`flux_score` stays UNSCALED.** It's already bounded `[0,1]` by
construction (a noisy-OR output -- see
`progress/flux_score_timeseries_findings.md`), so z-scoring it would obscure an
already-understood, already-saturated (0.85-0.96 mean) residual rather than
clarify it. Implemented by fitting the `StandardScaler` only on the
non-`flux_score` columns, transforming only those, and copying
`flux_score` through raw/untouched into the final scaled matrix
(`lstm/dataset.py::build_ticker_dataset`). A **module-load-time assertion**
(`lstm/dataset.py`, right after `VARIANTS` is defined) checks that
`TARGET_COL` is never in the unscaled-column set and that its position among
the scaled columns matches its position in `feature_cols` -- both are true
by construction today (`daily_return` is always first, `flux_score`, if
present, is always last), but a future column-order change would now fail
loudly at import time instead of silently mis-inverse-transforming
predictions.

Per-ticker (not global-across-tickers) scaling remains necessary: the 5
tracked tickers still have very different return/volatility profiles, so a
single global scaler would be dominated by the highest-volatility name.

## Per-ticker or shared model: **shared model with a ticker embedding** (UNCHANGED by R1)

Same reasoning as before R1 -- see the rest of this section preserved from
the original design: 291 train windows per ticker alone is too few for a
2-layer LSTM; pooling gives the shared recurrent core 1,455 train windows
(5x) with a small (dim=4) ticker embedding concatenated only at the head.
Not re-ablated under R1 either.

## Architecture and hyperparameter choices (UNCHANGED by R1, docstring-only update)

`PriceLSTM`'s architecture is byte-for-byte unchanged by R1 -- only
`n_features` (4/5 instead of 5/6, since `adj_close` was removed from the
feature vector) and the output's documented meaning (`daily_return`, not
`adj_close`) changed. Full table, restated from before R1 (nothing in it
changed):

| choice | value | reasoning |
|---|---|---|
| LSTM #1 | 64 units, full sequence output | matches spec_sheet.md Paper 1's stated architecture |
| Dropout | 0.2, after LSTM #1 only | matches Paper 1's placement exactly |
| LSTM #2 | 32 units, final hidden state only | matches Paper 1's second-layer width |
| Ticker embedding | dim=4, concatenated at the Dense head only | new to this implementation, unchanged by R1 |
| Dense output | 1 unit, linear | matches both papers |
| Loss | MSE | matches both papers; still appropriate for a standardized-return regression target (verified by reading `lstm/train.py` directly under R1 -- no price-level assumption was found anywhere in the training loop, so no logic change was needed there) |
| Optimizer | Adam, lr=1e-3 | Adam matches both papers; lr is an explicit sane default (neither paper states a numeric value) |
| Batch size | 32 | matches Paper 1's stated value |
| Max epochs | 150, with early stopping (patience=15) | actually implements what Paper 1's own prose claims but its own code never does |
| Gradient clipping | max_norm=1.0 | deviation from both papers, for numerical stability |
| Weight decay / L2 | none | neither paper uses it |
| Seed | 42 | fixed for reproducibility |

## Training dynamics / overfitting (R1 run, 2026-07-23)

Both variants again show the expected small-dataset overfitting shape: train
MSE (standardized-return scale) decreases while val MSE bottoms out early
and then drifts back up, with early stopping halting near the actual val
minimum:

**baseline** (best epoch 17 of 32 run, patience=15 triggered):

| epoch | train_mse (scaled) | val_mse (scaled) |
|---|---|---|
| 1 | 1.0674 | 1.0460 |
| 5 | 0.9512 | 0.9698 |
| 10 | 0.9407 | 0.9576 |
| **17 (best)** | **0.8795** | **0.9350** |
| 20 | 0.8524 | 0.9895 |
| 25 | 0.7944 | 1.0052 |
| 32 (stopped) | 0.7136 | 1.0993 |

**flux** (best epoch 12 of 27 run, patience=15 triggered):

| epoch | train_mse (scaled) | val_mse (scaled) |
|---|---|---|
| 1 | 0.9595 | 0.9762 |
| 5 | 0.9491 | 0.9670 |
| 10 | 0.9382 | 0.9516 |
| **12 (best)** | **0.9224** | **0.9459** |
| 15 | 0.8891 | 0.9658 |
| 20 | 0.8309 | 1.0018 |
| 27 (stopped) | 0.7486 | 1.0561 |

Note the scaled-MSE values are near 1.0 at epoch 1 (a `StandardScaler`
target starts near unit variance by construction, unlike the old
`MinMaxScaler`-on-price-level target, which started near ~0.03-0.08) --
not comparable in absolute magnitude to the pre-R1 table in this README's
git history; only the shape (train falls, val bottoms out, then rises) is
the meaningful comparison. Full per-epoch history:
`lstm/models/baseline_history.json`, `lstm/models/flux_history.json`.

## Results (R1 run, real, verified against the live DB, 2026-07-23)

**MAE/RMSE are now the PRIMARY headline metrics**, on raw `daily_return`
units (e.g. `0.02` = 2%/day) -- not raw price units. MAPE is still reported
for continuity but is demoted to a secondary metric; see "MAPE-on-returns
caveat" below for why.

### Test set (the number that matters for the headline comparison)

| ticker | baseline MAE | baseline RMSE | baseline MAPE | flux MAE | flux RMSE | flux MAPE | flux_score effect (by MAE) |
|---|---|---|---|---|---|---|---|
| NVDA | 0.020567 | 0.025481 | 206.452% | 0.019758 | 0.024657 | 174.335% | helped |
| TSM | 0.024103 | 0.030710 | 319.707% | 0.023277 | 0.029897 | 143.307% | helped |
| AMD | 0.042408 | 0.052922 | 188.714% | 0.040676 | 0.052006 | 123.386% | helped |
| ASML | 0.028977 | 0.035903 | 129.399% | 0.028976 | 0.035716 | 128.354% | ~flat (helped, negligibly) |
| INTC | 0.047903 | 0.062675 | inf% | 0.046419 | 0.061872 | inf% | helped |
| **average** | **0.032791** | **0.041538** | **inf%** | **0.031821** | **0.040830** | **inf%** | **helped, uniformly across all 5 tickers this time** |

(INTC's MAPE is `inf` because at least one INTC test day has an exactly
zero realized `daily_return`, dividing by zero in the MAPE formula -- see
"MAPE-on-returns caveat" below; this is expected, not a bug.)

### Persistence baselines (new, R1 -- model-free, computed directly from raw test data)

| baseline | MAE | RMSE | MAPE |
|---|---|---|---|
| predict return = 0 | 0.031617 | 0.043102 | nan% (0/0 on exact-zero-return days) |
| predict return = yesterday's realized return | 0.046913 | 0.061012 | inf% |
| **baseline LSTM** | **0.032791** | **0.041538** | inf% |
| **flux LSTM** | **0.031821** | **0.040830** | inf% |

**Honest reading: on the test window, neither LSTM variant clearly beats the
trivial "predict return = 0" baseline on MAE.** The baseline LSTM
(0.032791) is slightly *worse* than the zero-baseline (0.031617); the flux
LSTM (0.031821) is essentially tied with it (worse by ~0.0002, a difference
far smaller than plausible sampling noise on 380 test points). Both LSTM
variants clearly beat the naive "return = yesterday's return" baseline
(0.0469), which is the *worse* of the two trivial baselines here (return
series don't have strong positive day-to-day autocorrelation, so persistence
overshoots). On val (0.023207/0.023413 vs. zero-baseline's 0.023705) and
train (0.021335/0.021643 vs. 0.022024) the LSTM does modestly beat the zero
baseline -- i.e., the model shows some real signal in-sample/near-sample
that erodes to roughly nothing by the test window, consistent with this
project's already-documented overfitting + regime-shift pattern (see
"Known gaps" below). **Stated plainly, not spun: this run does not provide
strong evidence that either LSTM variant earns its keep over trivially
predicting zero return on this specific test window.**

### Train / val (for reference)

| split | variant | avg MAE | avg RMSE | avg MAPE |
|---|---|---|---|---|
| train | baseline | 0.021335 | 0.030129 | inf% |
| train | flux | 0.021643 | 0.030933 | inf% |
| val | baseline | 0.023207 | 0.031253 | 220.790% |
| val | flux | 0.023413 | 0.031441 | 144.353% |
| test | baseline | 0.032791 | 0.041538 | inf% |
| test | flux | 0.031821 | 0.040830 | inf% |

Full per-ticker per-split breakdown: `lstm/models/run_summary.json`,
`lstm/models/baseline_metrics.json`, `lstm/models/flux_metrics.json`.

### MAPE-on-returns caveat (new section, R1)

MAPE divides by `y_true`. On the OLD price-level target, `adj_close` is
never near zero, so MAPE was well-behaved. On the NEW `daily_return` target,
returns cluster near zero by their nature, and a handful of near/exact-zero
actual-return test days make MAPE blow up (`inf`, or `nan` on an exact 0/0)
-- this happened for real on this run (INTC's test MAPE is `inf`; the
overall test/train averages are `inf` because at least one ticker/split hits
this). **This is a real statistical property of applying MAPE to a return
target, not a bug being hidden** -- `lstm/evaluate.py`'s
`compute_metrics`/`recompute_metrics_independently` both handle it
explicitly (matching IEEE-754 float division semantics: `x/0 -> inf`,
`0/0 -> nan`) rather than crashing, and the resulting `inf`/`nan` values are
reported as-is, not silently dropped or clipped. This is exactly why MAE/RMSE
(scale-dependent but not divide-by-near-zero-prone) are now the PRIMARY
headline metrics for this target, with MAPE kept only for continuity/
secondary reference.

### Honest verdict on `flux_score` (R1 run)

**On this run, `flux_score` helped on every one of the 5 tickers' test MAE**
(NVDA -3.9%, TSM -3.4%, AMD -4.1%, ASML -0.003% (~flat), INTC -3.1%
relative), and pooled test MAE improved by -0.000970 (baseline 0.032791 ->
flux 0.031821). This is a *cleaner, more uniform* result than the pre-R1
run (where flux helped 2/5 tickers and hurt 3/5 on MAPE) -- but the
absolute effect size is small, both variants sit close to the trivial
zero-return baseline on this same test window (see above), and this is a
single 76-day test window with no walk-forward validation (see "Known
gaps"). Read as a mild, directionally consistent positive signal for
`flux_score`, not proof of a robust effect -- exactly the kind of question
R5 (walk-forward validation, a later step) exists to actually answer.

## Gate check result (R1's actual point, stated plainly)

**R1's falsifiable question**: did reframing the target from price level to
return, and the input features from raw price to return/ratio quantities,
break the prior run's "3-4 of 5 tickers are 100%-short every single test
day" pathology?

**Per-ticker long/short test-day counts (76 test days/ticker), regenerated
from the retrained checkpoints via `backtest.predictions.generate_test_predictions`
+ `backtest.strategy.signal_from_predictions` (independently re-run by this
session, not read from a cached file):**

| ticker | OLD (price-level) baseline short/long | OLD flux short/long | NEW (R1) baseline short/long | NEW (R1) flux short/long |
|---|---|---|---|---|
| NVDA | 36 / 40 | 41 / 35 | 43 / 33 | 43 / 33 |
| TSM | **76 / 0** (100% short) | **76 / 0** (100% short) | 49 / 27 | 43 / 33 |
| AMD | 69 / 7 | 67 / 9 | 49 / 27 | 55 / 21 |
| ASML | **76 / 0** (100% short) | **76 / 0** (100% short) | 59 / 17 | 69 / 7 |
| INTC | **76 / 0** (100% short) | **76 / 0** (100% short) | 28 / 48 | 49 / 27 |

**Verdict: the all-one-direction (100%-short) pathology is FIXED.** Not one
of the 10 (ticker x variant) position series is monolithic anymore -- every
ticker now has the model going long on at least 7 of 76 test days, and 3 of
5 tickers (NVDA, AMD both variants; TSM both variants; INTC baseline) are
within a reasonable distance of a 50/50 split. This is a real, structural
change in model behavior, not a coincidence of one run: the mechanism
matches the hypothesis stated above -- a return/ratio-based feature vector
does not get pinned to one side of a fixed scaled range the way a raw
extrapolating price level did.

**But the fix is partial, not complete, and should not be oversold**:
- `flux`/ASML is still heavily short-skewed (69/76 = 90.8% short) -- much
  milder than 100%, but the same directional bias in a smaller dose. This
  matches the "not a guarantee" caveat stated above:
  `price_to_sma_long`/`sma_ratio` can still drift outside their train range
  on a strong enough sustained rally, just less catastrophically than raw
  price level did.
- The pooled backtest is still net-negative on this test window: baseline
  cum. return **-6.995%**, flux **-5.010%** (both net of the same 10bps
  turnover / 2bps/day short-borrow cost model as before), vs. buy-and-hold
  **+71.283%** and SMA(10/50) crossover **+26.609%** on the identical
  window/cost model (both naive controls are literally unchanged code paths
  and reproduce the exact same numbers as the pre-R1 backtest, confirming
  R1 didn't accidentally touch anything outside the LSTM/prediction/signal
  pipeline). Pooled Sharpe: baseline -0.443, flux -0.196 -- both still
  negative, but a large, real improvement from the pre-R1 run's -2.77 /
  -2.53 (i.e., an order-of-magnitude smaller loss, not just a smaller
  percentage).
- **This is genuinely a large, real improvement (cum. loss cut from ~-34%
  to ~-6%, an ~80% reduction in the size of the loss; Sharpe improved by
  roughly 2.3-2.6 units), but it is still a losing strategy net of costs on
  this one test window, dominated by both naive controls.** R1's target
  reframing addressed the mechanism identified in Phase G's audit (the
  100%-one-direction extrapolation bias) and the position-balance/backtest
  numbers confirm that mechanism is substantially, not fully, defused --
  it did NOT, on its own, turn this into a profitable strategy. Whether the
  remaining gap is closed by R2 (cross-sectional/dollar-neutral
  construction, which changes how badly a directionally-wrong call on one
  name hurts the whole book) or R5 (walk-forward validation, to see if this
  test window's still-strong rally is simply too hard a regime for any
  return-based directional signal) is exactly what those later steps exist
  to test -- not concluded here.

Full backtest detail (per-ticker Sharpe/drawdown/turnover for the new run,
naive-control results, random-signal spread) is in `backtest/README.md`
(also updated as part of this task, its "R1 rerun addendum" section).

## Verification performed (this project's established discipline)

1. **Leakage check, unchanged logic, re-run and re-verified after the
   rework**: `lstm.dataset.verify_no_leakage` -- **exact same window counts
   as before R1**: `{'windows_checked': {'train': 1455, 'val': 375, 'test':
   380}, 'status': 'no leakage found'}` for both variants, confirming the
   feature/target plumbing changes did not alter windowing/split-assignment
   logic (which R1 was explicitly not supposed to touch).
2. **`StandardScaler` independently spot-checked (new, R1)**: hand-computed
   NVDA's train-only `daily_return` mean/std directly from `features_daily`
   via a fresh pandas calculation, entirely bypassing `lstm/scaling.py`:
   `mean=0.0017071721502888745`, `std=0.032109366311650712` (n=351,
   `ddof=0`). Compared against the fitted `StandardScaler` from
   `lstm.dataset.build_ticker_dataset('NVDA', BASELINE_FEATURE_COLS)`:
   **identical to full float64 precision** (`hand_mean == scaler.mean_[0]`
   and `hand_std == scaler.std_[0]` both `True`, bitwise).
3. **Metrics re-derived twice, independently, including the new inf/nan edge
   case**: `compute_metrics` (vectorized numpy) vs.
   `recompute_metrics_independently` (plain-Python loop) both computed for
   every ticker/split/variant combination, asserted equal
   (`assert_metrics_agree`, extended under R1 to also handle
   `inf == inf` in addition to `nan == nan`, since a `daily_return` target
   can produce real `inf`/`nan` MAPE values that a price-level target never
   could) -- the full training run completed with zero assertion failures.
   A third, separate independent re-derivation was also run for this
   report: NVDA's baseline-variant test MAE/RMSE recomputed from scratch via
   `backtest.predictions.generate_test_predictions` (a different code path
   than `lstm/run_train.py::evaluate_split`) -- **MAE=0.020567, RMSE=0.025481,
   n=76**, matching `run_summary.json`'s stored value to 6 decimal places.
4. **Window counts cross-checked against `run_summary.json` directly** (not
   just trusted from console output): `{'train': 1455, 'val': 375, 'test':
   380}` for both variants, matching `feature_split_bounds`
   (`features/README.md`)'s 291/75/76 per-ticker x 5 tickers exactly.
5. **`lstm/train.py` read directly, not assumed unaffected**: confirmed no
   price-level-specific logic exists anywhere in the training loop (loss
   function, optimizer, early stopping, checkpointing are all
   target-representation-agnostic) -- no code change was needed there, per
   the task's own instruction to verify rather than assume.

## Known gaps / limitations (be honest about these, don't round up)

- **The gate check's headline finding, restated**: R1 fixed the
  100%-one-direction pathology (see "Gate check result" above) but did NOT
  turn either LSTM variant into a profitable strategy on this test window --
  both are still net-negative, still dominated by buy-and-hold and the naive
  SMA crossover. The improvement is real (an ~80% reduction in the size of
  the pooled loss) but partial.
- **Neither LSTM variant clearly beats the trivial "predict return = 0"
  baseline on test-window MAE** (see "Persistence baselines" above) -- this
  is a more demanding, and more honest, bar than the pre-R1 README could
  even compute (that baseline was flagged as a missing gap when the target
  was still a price level). The model shows real, if modest, skill on
  train/val that erodes to roughly nothing by the test window.
- **Regime shift is still present, just less catastrophic than before.**
  `price_to_sma_long`/`sma_ratio` are hypothesized (not proven) to be more
  regime-stable than raw price level -- the ASML flux-variant's still
  90.8%-short skew on the test window is direct evidence that this
  hypothesis is not a complete fix, only a substantial mitigation.
- **Single train/val/test split, no walk-forward/rolling-origin
  backtesting** -- this was the limitation flagged here before R1/R2; R5
  (see the addendum at the top of this file, and `backtest/README.md`'s R5
  addendum) has now addressed it with a 4-fold expanding-window check. The
  headline single-window numbers throughout the rest of this README
  (R1-R4) are UNCHANGED by R5 -- they remain exactly what happened on the
  one 2026-04-02..2026-07-22 window; R5 is a separate, additional piece of
  evidence layered on top, not a replacement of these numbers.
- **Small dataset, restated**: still bottlenecked to 1,455 pooled train
  windows by `flux_scores_daily`'s 2024-07-19 start date -- unchanged by
  R1 (this is a data-coverage limit, not a target/feature framing issue).
- **Shared-model design, hyperparameters, and gradient clipping are still
  not re-ablated** under the new target -- R1's scope was target/feature
  reframing only, not a hyperparameter search; the same architecture that
  was tuned (informally, via the reasoning in "Architecture and
  hyperparameter choices" above) for the old price-level target is reused
  as-is for the new return target. A dedicated hyperparameter pass for the
  new target is a reasonable follow-up, not performed here.
- **CPU-only training environment**, unchanged from before R1.
- **MAPE-on-returns is genuinely noisier than MAPE-on-price-level was** --
  see the dedicated caveat section above; this is a property of the new
  target's statistics, not a code defect, but it does mean MAPE numbers
  in this README (and any future comparison to the pre-R1 numbers) are not
  apples-to-apples with the old price-level MAPE figures.

## Running it

```
./.venv/bin/python -m lstm.run_train
```

Trains both variants (baseline, then flux) end to end against the live
`data/events.db`, prints per-epoch train/val loss, runs the leakage check,
evaluates all three splits per ticker, computes the two persistence
baselines (new, R1), saves checkpoints + scaler stats + per-epoch history +
metrics JSON under `lstm/models/`, and prints a final
baseline-vs-flux-vs-persistence test-set comparison (MAE-led, with an
explicit UNDEFINED marker when MAPE hits inf/nan rather than a misleading
number).

### Real run performed (2026-07-23, R1 rework)

Command actually executed against the live `data/events.db`:
```
./.venv/bin/python -m lstm.run_train
```
Exit: both variants trained successfully (baseline stopped at epoch 32, best
epoch 17; flux stopped at epoch 27, best epoch 12), no errors, no assertion
failures in the independent metric-recompute check (including the new
inf/nan-aware `assert_metrics_agree` path, exercised for real by this run's
INTC test-set MAPE hitting `inf`). Full console output and all metrics
quoted in this README are taken directly from that run -- see
`lstm/models/run_summary.json` for the complete machine-readable record.

## Files

- `lstm/dataset.py` -- `Window` dataclass, per-ticker windowing
  (`build_ticker_dataset`), pooled multi-ticker dataset construction
  (`build_pooled_dataset`), the independent leakage check
  (`verify_no_leakage`, unchanged by R1, gained an optional `fold_id` R5
  parameter), and (new, R1) the derived `price_to_sma_long`/`sma_ratio`
  feature computation plus the scaled/unscaled-column assertions. (New, R5)
  `split_boundaries` parameter on `build_ticker_dataset`/`build_pooled_dataset`
  and the `split_for_date()` helper -- see R5 addendum at the top of this file.
- `lstm/walk_forward_bounds.py` (new, R5) -- `walk_forward_bounds` DB table
  (per-fold train/val/test date-range bounds), `write_fold_bounds_for_tickers`,
  `read_bounds_frame`, `verify_fold_boundaries`.
- `lstm/scaling.py` -- from-scratch `StandardScaler` (new, R1, used by
  `dataset.py`) and `MinMaxScaler` (kept, no longer used by this package but
  still a valid general-purpose class).
- `lstm/model.py` -- `PriceLSTM`: LSTM(64) -> Dropout(0.2) -> LSTM(32) ->
  concat ticker embedding(4) -> Dense(1). Architecture unchanged by R1;
  docstring updated to describe the standardized-return output.
- `lstm/train.py` -- `TrainConfig`, `train_model` (training loop: device
  management, train/eval mode switching, gradient clipping, early stopping,
  best-checkpoint restore, fixed seed). Unchanged by R1 (verified, not
  assumed -- see "Verification performed" above).
- `lstm/evaluate.py` -- `compute_metrics` / `recompute_metrics_independently`
  / `assert_metrics_agree` (MAE/RMSE/MAPE, computed two independent ways,
  now inf/nan-aware) and (new, R1) `persistence_baseline_metrics`.
- `lstm/run_train.py` -- CLI entry point (`python -m lstm.run_train`),
  orchestrates both variants, saves models + metrics, and (new, R1) computes
  + reports the persistence baselines.
- `lstm/models/` -- saved checkpoints (`baseline.pt`, `flux.pt`, each
  containing model weights, per-ticker `StandardScaler` stats, config, and
  metadata), per-epoch history (`*_history.json`), per-variant metrics
  (`*_metrics.json`), and the combined `run_summary.json` (now also
  containing a top-level `persistence_baselines` key).
