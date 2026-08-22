"""
Price ingestion orchestrator (Phase F, price half).

Run with:
    python -m price.run_ingest

Fetches maximum-available daily OHLCV history from yfinance for every ticker
in `config.mvp_scope.TRACKED_TICKERS`, stores it in `data/events.db`
(table `daily_prices`, see price/storage.py), and prints a per-ticker summary
(rows fetched / newly inserted / deduped-away / date range / error).

A ticker "failing" means yfinance could not return usable data for it at all
(network error, rate limit, empty response, unexpected schema) -- that is
reported as a real gap, never silently backfilled or interpolated. Exit code
is non-zero only if every single ticker failed outright.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

from config.mvp_scope import TRACKED_TICKERS
from src.price.fetch import fetch_all
from src.price.storage import PriceStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("price.run_ingest")


@dataclass
class TickerResult:
    ticker: str
    fetched: int = 0
    inserted: int = 0
    deduped: int = 0
    min_date: str | None = None
    max_date: str | None = None
    error: str | None = None


def run(period: str = "max") -> list[TickerResult]:
    store = PriceStore()
    fetch_results = fetch_all(TRACKED_TICKERS, period=period)

    results: list[TickerResult] = []
    for fr in fetch_results:
        tr = TickerResult(ticker=fr.ticker)
        if fr.error is not None:
            tr.error = fr.error
            results.append(tr)
            continue

        tr.fetched = len(fr.bars)
        if fr.bars:
            attempted, inserted = store.insert_bars(fr.bars)
            tr.inserted = inserted
            tr.deduped = attempted - inserted
        tr.min_date, tr.max_date = store.date_range(fr.ticker)
        results.append(tr)

    store.close()
    return results


def print_summary(results: list[TickerResult]) -> None:
    print("\n=== Price ingestion summary ===")
    header = (
        f"{'ticker':8} {'fetched':>8} {'inserted':>9} {'deduped':>8} "
        f"{'min_date':>10} {'max_date':>10}  error"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        err = r.error or ""
        print(
            f"{r.ticker:8} {r.fetched:8d} {r.inserted:9d} {r.deduped:8d} "
            f"{r.min_date or '-':>10} {r.max_date or '-':>10}  {err}"
        )
    print()


def main() -> int:
    results = run()
    print_summary(results)

    all_failed = all(r.error is not None for r in results)
    if all_failed:
        logger.error("Every ticker failed -- exiting non-zero.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
