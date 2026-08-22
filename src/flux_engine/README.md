# Flux Engine (Phase E)

Implements `progress/flux_formula.md` exactly: combines Phase B's classified events
(severity, confidence, provenance) with Phase D's exposure graph and decay
functions into a single `flux_score(stock, t) ∈ [0,1]`, plus an optional
signed `direction` where real polarity data exists.

## Files
- `formula.py` — the pure scoring math (`event_contribution`, `flux_score`,
  `flux_scores_for_stocks`). Zero DB/IO dependency, so it can be verified in
  isolation against the design doc's worked example (see below).
- `query.py` — loads real `(raw_events JOIN classified_events)` rows from
  `data/events.db` into the plain dicts `formula.py` expects. Always uses
  `raw_events.published_at` (never `ingested_at`), and always excludes
  `event_type = 'unclassified'` at the SQL level, per doc §2.
- `run.py` — CLI: `./.venv/bin/python -m flux_engine.run`. Computes
  `flux_score(s, now)` for every tracked stock from the live corpus.

## Correctness verification (real, not claimed)

`progress/flux_formula.md` §7 hand-derives a worked example (a Taiwan earthquake +
a US geopolitical GDELT event, evaluated for all 5 stocks). Running that exact
scenario through `formula.flux_scores_for_stocks()` reproduces every value to
within floating-point/rounding tolerance (<0.001):

| ticker | doc §7 expected | engine computed | match |
|---|---|---|---|
| TSM | 0.5088 | 0.5086 | yes |
| NVDA | 0.3877 | 0.3877 | yes |
| AMD | 0.3640 | 0.3639 | yes |
| ASML | 0.2180 | 0.2180 | yes |
| INTC | 0.1816 | 0.1816 | yes |

(Reproduce with the snippet in this repo's development history, or by
constructing the two doc §7 events as dicts and calling
`flux_scores_for_stocks`.)

## Real run against the live corpus (2026-07-19)

```
./.venv/bin/python -m flux_engine.run
```
loaded 104 non-`unclassified` events (74 `rule_based`, 30 `gdelt_derived`)
within the derived 34-day lookback window, and produced:

| ticker | flux_score | direction | coverage | n_events |
|---|---|---|---|---|
| INTC | 0.9599 | -0.735 | 0.760 | 53 |
| TSM | 0.8495 | -0.735 | 0.648 | 53 |
| NVDA | 0.8243 | -0.735 | 0.627 | 53 |
| AMD | 0.8144 | -0.735 | 0.646 | 53 |
| ASML | 0.7786 | -0.735 | 0.630 | 53 |

**These absolute numbers should NOT be read as "the market is about to move
80-96%."** They are a concrete, real demonstration of a risk the design doc
already flagged rather than a new bug:

- **Doc §5, judgment call #10 ("independence assumption... over-counts
  near-duplicate events")** is visibly happening. All 5 stocks are driven by
  53 contributing events, dominated by `geopolitical_military_tension` rows —
  inspecting the actual titles behind them shows most are **different articles
  about the same ongoing US-Iran conflict** ("US-Iran war highlights
  vulnerability of Mideast undersea cables", "Iran supreme leader vows
  'unforgettable lessons'...", "Dow Jones Futures: Iran Attack Kills Two U.S.
  Troops...", etc.) — real news, but heavily correlated, not ~50 independent
  risk factors. Noisy-OR (`1 - Π(1-cᵢ)`) is designed for independent factors;
  feeding it 15+ near-duplicate articles about one situation saturates the
  score toward 1.0 regardless of how each individual contribution is scaled.
- **A couple of genuine rule-labeler false positives are also visibly
  contributing**, e.g. a personal-finance HELOC article and a UNESCO
  climate-heritage piece both got tagged `geopolitical_military_tension`
  (stray keyword matches) — consistent with the ~72% (not 100%) spot-check
  accuracy already documented in `labeling/README.md`.
- The **relative ranking is still directionally sane** given the dominant
  driver is broad `United States`-origin tension rather than a Taiwan-specific
  flux: INTC (heaviest direct US manufacturing exposure) actually ranks
  highest here, not TSM — a genuinely different (and correctly derived) result
  from the Taiwan-earthquake worked example, not an inconsistency.

**Not a fix shipped in Phase E** (flagged for Phase F/G, matching the design
doc's own disposition of this exact issue): event de-duplication/clustering
before aggregation (e.g. cluster same-story articles within a short time
window and treat the cluster as one event for noisy-OR purposes) would be the
correct fix. Shipping that now would mean guessing at a clustering heuristic
with no calibration data, which is exactly the kind of unjustified addition
this project's discipline avoids — better to surface the real behavior
honestly (as this README does) than to quietly patch around it.

## Known deviation from the design doc

`progress/flux_formula.md` §6 flags an unresolved trading-day-vs-calendar-day
ambiguity inherited from Phase D. This implementation uses **calendar days**
(`(t - published_at).total_seconds() / 86400`) for `Δ`, the simpler of the two
options and the one that doesn't require pulling in a market-calendar
dependency before one is otherwise needed (Phase F/G). Documented in the doc
itself (§6) as the adopted v1 choice — switch to a real trading calendar once
one exists for the LSTM/backtest phases.

## Known limitations (inherited + new)
- Everything in `progress/flux_formula.md` §8's judgment-call table, especially
  #3/#4 (uncalibrated `rel` discounts), #10 (event over-counting, demonstrated
  above), and #13 (China not a graph node).
- `run.py` computes scores at `t = now`; it does not yet persist a time series
  of scores anywhere (no `flux_scores` table) — that's a natural Phase F/G
  addition once there's a backtest to feed.
