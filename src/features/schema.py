"""
Model-ready feature-row schema for the LSTM feature-building package.

One `FeatureRow` per (ticker, trading day), joining `daily_prices` (Phase F,
price half) against `flux_scores_daily` (Phase F, flux half). Mirrors
`price.schema.PriceBar`'s discipline: a dataclass with an explicit
`validate()` that raises rather than silently coercing/fabricating a value,
and a natural composite key `(ticker, date)` -- no synthetic hash id needed.

Date discipline: `date` is the trading-day calendar date (`YYYY-MM-DD`) taken
directly from `daily_prices.date` (see `price/schema.py`'s module docstring
for why that column carries no time-of-day / no UTC conversion). This module
does not reinterpret it.

No scaling/normalization happens here, deliberately -- see `features/README.md`
("Scaling is deferred"). Every numeric column in this table is the raw,
unscaled value. Fitting scalers only on the train partition is the LSTM
training step's job, not this build step's.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_VALID_SPLITS = {"train", "val", "test"}


class InvalidFeatureRowError(ValueError):
    pass


@dataclass
class FeatureRow:
    ticker: str                 # e.g. "NVDA", must be one of config.mvp_scope.TRACKED_TICKERS
    date: str                   # "YYYY-MM-DD", trading-day date, from daily_prices.date
    adj_close: float            # split/dividend-adjusted close (price level, unscaled)
    daily_return: float         # pct_change(adj_close, 1) vs the prior TRADING day, causal
    sma_short: float            # 10-trading-day trailing SMA of adj_close, causal
    sma_long: float             # 50-trading-day trailing SMA of adj_close, causal
    volatility_10: float        # 10-trading-day trailing rolling std of daily_return, causal
    flux_score: float          # flux_scores_daily.flux_score for this ticker/date
    flux_score_source_date: str  # calendar date the flux_score value actually came from
                                   # (== date unless a forward-fill fallback was needed; see build.py)
    split: str                  # "train" | "val" | "test" -- chronological assignment, see build.py
    computed_at: str = ""       # ISO 8601 UTC, when this row was computed

    def validate(self) -> None:
        """
        Cheap internal-consistency checks. Raises InvalidFeatureRowError
        rather than silently coercing/dropping/imputing a "fixed" value --
        a row that fails these checks indicates a real upstream problem
        (insufficient warmup history, a broken join, etc.) and should be
        surfaced, not patched.
        """
        if not self.ticker:
            raise InvalidFeatureRowError("FeatureRow.ticker is empty")
        if not _DATE_RE.match(self.date):
            raise InvalidFeatureRowError(f"FeatureRow.date {self.date!r} is not YYYY-MM-DD")
        if not _DATE_RE.match(self.flux_score_source_date):
            raise InvalidFeatureRowError(
                f"FeatureRow.flux_score_source_date {self.flux_score_source_date!r} "
                "is not YYYY-MM-DD"
            )
        if self.flux_score_source_date > self.date:
            # The whole point of this field: a flux_score attributed to row `date`
            # must never come from a date strictly after `date` -- that would be
            # future information leaking into a "current" feature.
            raise InvalidFeatureRowError(
                f"FeatureRow leakage guard failed for {self.ticker} {self.date}: "
                f"flux_score_source_date={self.flux_score_source_date} is AFTER date={self.date}"
            )
        for field_name in ("adj_close", "daily_return", "sma_short", "sma_long",
                           "volatility_10", "flux_score"):
            value = getattr(self, field_name)
            if value is None or value != value:  # None or NaN
                raise InvalidFeatureRowError(
                    f"FeatureRow.{field_name} is missing/NaN for {self.ticker} {self.date}"
                )
        if self.adj_close < 0:
            raise InvalidFeatureRowError(
                f"FeatureRow.adj_close is negative for {self.ticker} {self.date}"
            )
        if self.split not in _VALID_SPLITS:
            raise InvalidFeatureRowError(
                f"FeatureRow.split {self.split!r} not one of {_VALID_SPLITS} "
                f"for {self.ticker} {self.date}"
            )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SplitBounds:
    """
    One row per (ticker, split), recording the chronological date-range
    boundaries actually used to assign `FeatureRow.split`. Stored alongside
    `features_daily` so the split is unambiguous and reproducible from the
    DB alone, without re-deriving it from row order.
    """
    ticker: str
    split: str              # "train" | "val" | "test"
    start_date: str
    end_date: str
    n_rows: int
    lookback_days: int      # the 60-trading-day window size these bounds assume
    first_usable_label_date: Optional[str]
    # first date in this split that can serve as a WINDOW LABEL (i.e. has
    # >= lookback_days trading days of feature history available on or before
    # it, counting back through earlier splits if needed -- see build.py).
    # None if no date in this split has enough history yet.
    n_usable_labels: int
    computed_at: str = ""
