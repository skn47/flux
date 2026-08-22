#!/usr/bin/env bash
# Manual, one-command refresh of the full pipeline, in dependency order.
# Ingestion (step 1 below) can also run on its own schedule via
# ingestion/scheduler.py, but everything from step 2 onward is still
# manual-only (see api/README.md's "Data freshness" section) -- run this
# before a demo, or whenever you want every table the API reads to reflect
# the latest ingested data.
#
# Steps 5/6 (2026-08-16): flux_engine.run_timeseries does a full historical
# re-cluster every run, which can promote previously-non-winning gdelt_derived
# rows into event_catalog as fresh cluster winners -- rows that have never
# been through labeling/run_gdelt_reclassify.py's Stage 1/2 content check, so
# they can carry the same QuadClass-only false positives that check exists to
# catch (see labeling/README.md's GDELT section). Before this was wired in
# here, nothing ever re-ran that check after a routine pipeline refresh, so
# the backlog could silently regrow. This is deliberately a SINGLE bounded
# pass, not a full convergence loop: the historical round-by-round data decays
# slowly (~20%/round recently), so fully draining the backlog to zero would
# take 10+ rounds of local-LLM (Ollama) time -- too expensive for a script
# meant to run before a demo. One pass catches the bulk of each run's fresh
# promotions (a single round has historically corrected ~88%+ of what it
# checks) without an unbounded runtime. See labeling/README.md for the full
# tradeoff writeup and how to run further manual rounds if you want the
# residual pushed down further before something high-stakes.
#
# Step 7 (2026-08-19): same gap, different table -- narratives/
# run_precompute_narratives.py (event_narratives, feeds the 3D causality
# graph's per-event AI explanation) was never wired into this script either,
# so every run_timeseries rebuild above could promote new events into a
# ticker's top-5 daily attribution with no narrative generated for them,
# silently regrowing that gap on every refresh (same shape as the step 5/6
# fix, just never caught for this table at the time). Placed after both
# run_timeseries reruns since it depends on the final flux_attribution_daily
# state, and it's already idempotent/resumable by construction (only ever
# processes event_ids not yet in event_narratives), so no bounding needed
# here the way step 5 needed one.
set -e

cd "$(dirname "$0")/.."

echo "=== 1/11: ingestion.run_ingest ==="
./.venv/bin/python -m src.ingestion.run_ingest

echo "=== 2/11: labeling.run_label ==="
./.venv/bin/python -m src.labeling.run_label

echo "=== 3/11: price.run_ingest ==="
./.venv/bin/python -m src.price.run_ingest

echo "=== 4/11: flux_engine.run_timeseries (also populates event_catalog) ==="
./.venv/bin/python -m src.flux_engine.run_timeseries

echo "=== 5/11: labeling.run_gdelt_reclassify (content re-check, single pass) ==="
./.venv/bin/python -m src.labeling.run_gdelt_reclassify

echo "=== 6/11: flux_engine.run_timeseries (re-rebuild after step 5's corrections) ==="
./.venv/bin/python -m src.flux_engine.run_timeseries

echo "=== 7/11: narratives.run_precompute_narratives (AI explanations for newly-promoted top-5 events) ==="
./.venv/bin/python -m src.narratives.run_precompute_narratives

echo "=== 8/11: features.run_build ==="
./.venv/bin/python -m src.features.run_build

echo "=== 9/11: lstm.run_train ==="
./.venv/bin/python -m src.lstm.run_train

echo "=== 10/11: backtest.persist_predictions ==="
./.venv/bin/python -m src.backtest.persist_predictions

echo "=== 11/11: backtest.persist_seq2seq_forecast ==="
./.venv/bin/python -m src.backtest.persist_seq2seq_forecast

echo "=== Done. api/db.py opens a fresh read-only connection per request (WAL mode), so the running API server picks up this refresh on its next request -- no restart needed. ==="
