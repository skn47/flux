"""
CLI: compute current flux_score(s, t=now) for every tracked stock from the
real, live data/events.db corpus. Run:  ./.venv/bin/python -m flux_engine.run
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config.mvp_scope import TRACKED_TICKERS
from src.propagation.graph import ExposureGraph
from src.propagation.decay import DEFAULT_HALF_LIFE_DAYS
from src.flux_engine.formula import flux_scores_for_stocks, lookback_window_days
from src.flux_engine.query import load_events


def main() -> None:
    t = datetime.now(timezone.utc)
    window_days = lookback_window_days()
    since = t - timedelta(days=window_days)

    events = load_events(since=since)
    print(f"Loaded {len(events)} non-unclassified events published within the "
          f"{window_days}-day lookback window (t={t.isoformat()}).")

    by_source: dict[str, int] = {}
    for e in events:
        by_source[e["label_source"]] = by_source.get(e["label_source"], 0) + 1
    print(f"By label_source: {by_source}\n")

    graph = ExposureGraph()
    results = flux_scores_for_stocks(events, TRACKED_TICKERS, t, graph=graph)

    print(f"{'ticker':6} {'flux_score':>11} {'direction':>10} {'coverage':>9} {'n_events':>9}")
    for ticker, r in sorted(results.items(), key=lambda kv: -kv[1].flux_score):
        direction_str = f"{r.direction:+.3f}" if r.direction is not None else "   n/a"
        print(f"{ticker:6} {r.flux_score:11.4f} {direction_str:>10} "
              f"{r.direction_coverage:9.4f} {len(r.contributing_events):9d}")

    print("\nTop contributing events per stock (up to 3 each):")
    for ticker, r in sorted(results.items(), key=lambda kv: -kv[1].flux_score):
        if not r.contributing_events:
            continue
        top = sorted(r.contributing_events, key=lambda se: -se.contribution)[:3]
        print(f"  {ticker}:")
        for se in top:
            print(f"    c={se.contribution:.4f} imp={se.imp:.3f} tau={se.tau:.3f} "
                  f"days_since={se.days_since:.1f} type={se.event_type} "
                  f"source={se.label_source} event_id={se.event_id[:12]}...")


if __name__ == "__main__":
    main()
