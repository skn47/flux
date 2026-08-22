"""
Causal (trailing, no look-ahead) technical indicators computed purely from
`daily_prices`. Kept deliberately lean per `progress/phase_f_lstm_decisions.md`
decision #2 (price/technical features + a single external signal, not a
macro/fundamental dump).

Every function here operates on a full per-ticker price history DataFrame
(sorted ascending by date) and returns trailing/rolling statistics. Pandas'
`.rolling(window)` and `.pct_change(1)` are both trailing by construction
(the value at row i only reads rows <= i) -- NOT centered -- so nothing here
can see a future row. This is exactly the causality the LSTM pipeline
requires: the feature vector for trading day t must only reflect information
available as of the close of day t.

Indicator choices and why:
  - `adj_close`: split/dividend-adjusted close, not raw `close`. Per
    `price/README.md`, raw `close` has artificial jumps at every split (e.g.
    NVDA's 10:1 split in 2024) that are not real price moves; using it would
    inject fake discontinuities into both the price level and every
    indicator derived from it.
  - `daily_return` = pct_change(adj_close, 1): the single-day log-free simple
    return. Included both as a feature in its own right and as the input to
    `volatility_10` below.
  - `sma_short` (10-day) / `sma_long` (50-day): a classic short/long moving
    average pair (same spirit as the 10/50 "golden/death cross" convention),
    giving the model both a fast and a slow read on trend without a large
    grid of extra windows.
  - `volatility_10`: 10-day trailing rolling std of `daily_return`, i.e.
    realized volatility -- a standard, cheap-to-justify measure of recent
    price turbulence.
  - Deliberately excluded: raw OHLV columns as separate features (open,
    high, low, volume), and any indicator needing a centered/non-causal
    window (e.g. a centered moving average or a lookahead-based volatility
    estimator). Kept lean per the decisions doc; see features/README.md for
    the full rationale of what was excluded and why.
"""

from __future__ import annotations

import pandas as pd

SMA_SHORT_WINDOW = 10
SMA_LONG_WINDOW = 50
VOLATILITY_WINDOW = 10

# Longest lookback any indicator here needs before its first non-NaN value.
MAX_WARMUP_TRADING_DAYS = max(SMA_SHORT_WINDOW, SMA_LONG_WINDOW, VOLATILITY_WINDOW + 1)


def compute_technical_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    `price_df` must be a single ticker's full price history, sorted
    ascending by `date`, with at least a `date` and `adj_close` column
    (extra columns are passed through untouched).

    Adds four causal columns: `daily_return`, `sma_short`, `sma_long`,
    `volatility_10`. Rows before enough trading-day history has accumulated
    (see MAX_WARMUP_TRADING_DAYS) will have NaN in one or more of these
    columns -- callers must either supply enough history before the window
    they actually need, or drop/flag NaN rows rather than silently using
    them (see `features/build.py`, which passes the FULL available price
    history so the warmup NaNs land safely before the flux_score-covered
    window and never reach the final output table).
    """
    if not price_df["date"].is_monotonic_increasing:
        raise ValueError("compute_technical_features requires price_df sorted ascending by date")

    out = price_df.copy()
    out["daily_return"] = out["adj_close"].pct_change(periods=1)
    out["sma_short"] = out["adj_close"].rolling(window=SMA_SHORT_WINDOW, min_periods=SMA_SHORT_WINDOW).mean()
    out["sma_long"] = out["adj_close"].rolling(window=SMA_LONG_WINDOW, min_periods=SMA_LONG_WINDOW).mean()
    out["volatility_10"] = (
        out["daily_return"].rolling(window=VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).std()
    )
    return out
