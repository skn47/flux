#!/usr/bin/env bash
# Weekly, separate-from-nightly regeneration of the two static Backtests-
# drawer artifacts (backtest/sector_metrics.json, backtest/walk_forward_
# results.json). Deliberately NOT part of scripts/refresh_pipeline.sh's
# nightly run: backtest.walk_forward retrains fresh models from scratch
# across 4 expanding-window folds (never warm-started), too costly to repeat
# every night. Both scripts write a `computed_at` timestamp into their JSON
# output, and api/routers/reports.py's cache is keyed on (path, mtime), so
# the API and frontend pick up a fresh run automatically -- no API restart
# needed, same as the nightly pipeline.
#
# Run:
#   ./scripts/refresh_backtests.sh
# Or via cron (see api/README.md's "Data freshness" section) -- wrapped in
# its own flock so it can never overlap with itself, same pattern as
# refresh_pipeline.sh's cron entry.
set -e

cd "$(dirname "$0")/.."

echo "=== 1/2: backtest.run_sector_backtest ==="
./.venv/bin/python -m src.backtest.run_sector_backtest

echo "=== 2/2: backtest.walk_forward ==="
./.venv/bin/python -m src.backtest.walk_forward

echo "=== Done. api/routers/reports.py picks this up on its next request (mtime-keyed cache) -- no restart needed. ==="
