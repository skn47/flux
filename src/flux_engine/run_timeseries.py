"""
CLI: compute and persist the daily flux_score(s, t) time series for every
tracked stock across the full backfilled history. Run:
    ./.venv/bin/python -m flux_engine.run_timeseries [--start YYYY-MM-DD] [--end YYYY-MM-DD]

See flux_engine/timeseries.py for why this bulk-loads once and pre-clusters
globally rather than calling flux_engine.formula.flux_score per date.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta, timezone

from config.mvp_scope import TRACKED_TICKERS
from src.propagation.graph import ExposureGraph
from src.flux_engine.query import load_events
from src.flux_engine.timeseries import (
    build_clusters,
    build_event_catalog,
    compute_daily_series,
    enrich_event_catalog_urls,
    store_daily_attribution,
    store_daily_series,
    store_event_catalog,
)

DEFAULT_START = date(2024, 7, 19)  # earliest real backfilled GDELT coverage


def main() -> None:
    p = argparse.ArgumentParser(description="Compute + store the daily flux_score time series")
    p.add_argument("--start", default=DEFAULT_START.isoformat())
    p.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    t0 = time.time()
    since = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    print(f"Bulk-loading events since {since.isoformat()} ...")
    events = load_events(since=since)
    print(f"Loaded {len(events)} non-unclassified events in {time.time()-t0:.1f}s")

    t1 = time.time()
    graph = ExposureGraph()
    clusters = build_clusters(events, graph=graph)
    print(f"Built {len(clusters)} clusters from {len(events)} events in {time.time()-t1:.1f}s")

    t1b = time.time()
    catalog = build_event_catalog(events, clusters)
    enrich_event_catalog_urls(catalog)
    n_catalog_rows = store_event_catalog(catalog, since=since)
    print(f"Stored {n_catalog_rows} rows into event_catalog in {time.time()-t1b:.1f}s")

    t2 = time.time()
    results = compute_daily_series(clusters, TRACKED_TICKERS, start, end)
    n_days = (end - start).days + 1
    print(f"Computed {n_days} days x {len(TRACKED_TICKERS)} stocks in {time.time()-t2:.1f}s")

    t3 = time.time()
    n_rows = store_daily_series(results)
    print(f"Stored {n_rows} rows into flux_scores_daily in {time.time()-t3:.1f}s")

    t4 = time.time()
    n_attr_rows = store_daily_attribution(results)
    print(f"Stored {n_attr_rows} rows into flux_attribution_daily in {time.time()-t4:.1f}s")
    print(f"Total elapsed: {time.time()-t0:.1f}s")

    print(f"\n{'ticker':6} {'mean':>8} {'min':>8} {'max':>8} {'std':>8}")
    for ticker in TRACKED_TICKERS:
        series = [d.flux_score for d in results[ticker]]
        n = len(series)
        mean = sum(series) / n
        variance = sum((x - mean) ** 2 for x in series) / n
        print(f"{ticker:6} {mean:8.4f} {min(series):8.4f} {max(series):8.4f} {variance**0.5:8.4f}")


if __name__ == "__main__":
    main()
