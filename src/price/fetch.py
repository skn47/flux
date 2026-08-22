"""
yfinance-backed daily OHLCV connector for the tracked tickers.

Pulls DAILY bars only (no intraday) for the maximum history yfinance will
serve for each ticker (`period="max"`), so the downstream LSTM has as much
lead-in history as possible for whichever lookback window (60 or 216 trading
days) gets chosen later. This module does no feature engineering, no
resampling, no gap-filling -- it stores exactly the bars yfinance returns.

Real data only: if a ticker's request fails (network error, delisting,
malformed response), that is recorded as a real per-ticker failure and
returned to the caller -- it is NEVER papered over with synthetic or
interpolated bars.

Timezone / date handling (see price/schema.py's module docstring for the
full rationale): yfinance's daily index is a midnight timestamp localized to
the listing exchange's local timezone. We take the calendar date directly
off that localized timestamp with NO conversion to UTC or any other
timezone, since these are trading-day labels, not point-in-time instants.
Verified empirically (2026-07-21) that all 5 tracked tickers -- including
the ADRs TSM and ASML -- report their yfinance index in `America/New_York`
(i.e. NYSE/NASDAQ local time), not their home-exchange timezone (Taiwan
Stock Exchange / Euronext Amsterdam). This is expected: yfinance/Yahoo
Finance serves these under their US-listed ADR tickers, and the bars ARE the
US trading session's bars, not a re-timestamped version of the home-exchange
session.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

from src.price.schema import InvalidPriceBarError, PriceBar

logger = logging.getLogger("price.fetch")

SOURCE_NAME = "yfinance"

# Columns required from yfinance's history() call. auto_adjust=False is
# deliberate: it's the only way yfinance gives us BOTH the raw close and a
# separately computed split/dividend-adjusted close in the same call. With
# the (newer) default auto_adjust=True, "Close" IS the adjusted close and
# there is no raw close column at all -- we'd lose information the spec
# asked for (both close and adj_close).
REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


class TickerFetchError(Exception):
    """A ticker's history could not be fetched/parsed at all."""


@dataclass
class TickerFetchResult:
    ticker: str
    bars: list[PriceBar] = field(default_factory=list)
    error: Optional[str] = None
    skipped_invalid_rows: int = 0


def fetch_ticker_history(ticker: str, period: str = "max") -> TickerFetchResult:
    """
    Fetch daily OHLCV history for one ticker. Raises TickerFetchError on a
    hard failure (network, empty/malformed response); individual malformed
    rows within an otherwise-good response are skipped and counted, not
    fabricated or dropped silently -- see result.skipped_invalid_rows.
    """
    ingested_at = datetime.now(timezone.utc).isoformat()
    result = TickerFetchResult(ticker=ticker)

    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=False, actions=True)
    except Exception as exc:  # noqa: BLE001 - network/parse errors from yfinance/curl_cffi
        raise TickerFetchError(f"{type(exc).__name__}: {exc}") from exc

    if df is None or df.empty:
        raise TickerFetchError(
            f"yfinance returned no rows for {ticker!r} (period={period!r})"
        )

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise TickerFetchError(
            f"yfinance response for {ticker!r} is missing expected columns: {missing_cols} "
            f"(got columns={list(df.columns)})"
        )

    if df.index.tz is None:
        raise TickerFetchError(
            f"yfinance response for {ticker!r} has a tz-naive index -- refusing to guess "
            f"which exchange timezone it belongs to."
        )

    seen_dates: set[str] = set()
    for ts, row in df.iterrows():
        # Deliberately NOT converting timezone -- see module docstring.
        date_str = ts.date().isoformat()

        if date_str in seen_dates:
            logger.warning(
                "price.fetch: %s has duplicate rows for trading day %s -- keeping "
                "first occurrence, skipping duplicate.",
                ticker,
                date_str,
            )
            result.skipped_invalid_rows += 1
            continue

        try:
            bar = PriceBar(
                ticker=ticker,
                date=date_str,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                adj_close=float(row["Adj Close"]),
                volume=int(row["Volume"]),
                source=SOURCE_NAME,
                ingested_at=ingested_at,
            )
            bar.validate()
        except (InvalidPriceBarError, ValueError, TypeError) as exc:
            logger.warning(
                "price.fetch: skipping invalid row for %s %s: %s", ticker, date_str, exc
            )
            result.skipped_invalid_rows += 1
            continue

        seen_dates.add(date_str)
        result.bars.append(bar)

    return result


def fetch_all(tickers: list[str], period: str = "max") -> list[TickerFetchResult]:
    """
    Fetch history for every ticker, catching per-ticker failures so one
    broken/rate-limited ticker can never take down the others (same
    discipline as ingestion.run_ingest.run_source).
    """
    results: list[TickerFetchResult] = []
    for ticker in tickers:
        try:
            result = fetch_ticker_history(ticker, period=period)
            logger.info(
                "%s -> %d bars (%d skipped invalid rows)",
                ticker,
                len(result.bars),
                result.skipped_invalid_rows,
            )
        except TickerFetchError as exc:
            logger.error("price.fetch: %s failed: %s", ticker, exc)
            result = TickerFetchResult(ticker=ticker, error=str(exc))
        results.append(result)
    return results
