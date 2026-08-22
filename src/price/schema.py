"""
Shared daily-bar schema for the price ingestion package (Phase F, price half).

One `PriceBar` per (ticker, trading day). Unlike `ingestion.schema.RawEvent`,
which needs a synthetic sha256 dedup id because news records have no natural
key, a daily OHLCV bar already has a natural, stable composite key:
`(ticker, date)`. No hash is manufactured here -- `storage.py` uses that pair
directly as the SQLite PRIMARY KEY, which is what makes re-running ingestion
idempotent (`INSERT OR IGNORE`).

Date discipline (read this before touching `date`):
  - `date` is a trading-day date string in `YYYY-MM-DD` form -- a calendar
    day, not an instant in time. It deliberately carries NO time-of-day and
    NO UTC conversion.
  - yfinance returns each daily bar's index as a timestamp localized to the
    listing exchange's local timezone (`America/New_York` for all 5 tracked
    tickers -- verified empirically, see price/README.md), always at
    midnight local time (00:00:00-04:00 or -05:00 depending on DST). That
    midnight timestamp is a *label* for "this is the bar for trading day X,"
    not a real point-in-time event. Converting it to UTC before taking the
    date (e.g. `.tz_convert("UTC").date()`) would shift it into the previous
    calendar day for roughly half the year (whenever the UTC offset is
    negative), silently mislabeling every bar. `fetch.py` therefore takes
    `.date()` directly off the exchange-local timestamp, WITHOUT any tz
    conversion. This mirrors the project's existing discipline of never
    silently reinterpreting a source's native semantics (see the FIPS vs.
    CAMEO country-code callout in `config/mvp_scope.py`).
  - `close` is the raw (unadjusted) closing price as printed on the ticker
    tape; `adj_close` is yfinance's dividend/split-adjusted close. Both are
    stored side by side deliberately -- see price/README.md for why a
    feature pipeline should almost always prefer `adj_close` and why we
    don't collapse the two into one column here (that's a downstream
    feature-engineering decision, out of scope for this ingestion stage).

This module does no fetching, no scoring, no feature engineering -- it only
defines the shape of a bar and a light validator.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class InvalidPriceBarError(ValueError):
    pass


@dataclass
class PriceBar:
    ticker: str        # e.g. "NVDA", must be one of config.mvp_scope.TRACKED_TICKERS
    date: str           # "YYYY-MM-DD", trading-day date, exchange-local, no time-of-day
    open: float
    high: float
    low: float
    close: float        # raw/unadjusted close
    adj_close: float     # split/dividend-adjusted close
    volume: int
    source: str = "yfinance"
    ingested_at: str = ""  # ISO 8601 UTC, when this run pulled the record

    def validate(self) -> None:
        """
        Cheap internal-consistency checks. Raises InvalidPriceBarError rather
        than silently coercing/fabricating a "fixed" value -- a bar that
        fails these checks should be dropped and logged by the caller, not
        patched.
        """
        if not self.ticker:
            raise InvalidPriceBarError("PriceBar.ticker is empty")
        if not _DATE_RE.match(self.date):
            raise InvalidPriceBarError(
                f"PriceBar.date {self.date!r} is not YYYY-MM-DD"
            )
        for field_name in ("open", "high", "low", "close", "adj_close"):
            value = getattr(self, field_name)
            if value is None or value != value:  # None or NaN
                raise InvalidPriceBarError(
                    f"PriceBar.{field_name} is missing/NaN for {self.ticker} {self.date}"
                )
            if value < 0:
                raise InvalidPriceBarError(
                    f"PriceBar.{field_name} is negative ({value}) for {self.ticker} {self.date}"
                )
        if self.volume is None or self.volume < 0:
            raise InvalidPriceBarError(
                f"PriceBar.volume invalid ({self.volume}) for {self.ticker} {self.date}"
            )
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise InvalidPriceBarError(
                f"PriceBar OHLC inconsistent for {self.ticker} {self.date}: "
                f"open={self.open} high={self.high} low={self.low} close={self.close}"
            )

    def as_dict(self) -> dict:
        return asdict(self)
