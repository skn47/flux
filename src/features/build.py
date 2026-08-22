"""
Build per-ticker, per-trading-day model-ready feature tables for the LSTM,
joining `daily_prices` (authoritative trading-day calendar) against
`flux_scores_daily` (daily calendar-day flux signal), per the design in
`progress/phase_f_lstm_decisions.md`.

No scaling happens here (see features/README.md, "Scaling is deferred").
This module only produces raw, unscaled `FeatureRow`s plus a chronological
train/val/test split assignment and its date-range boundaries.
"""

from __future__ import annotations

import bisect
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config.mvp_scope import TRACKED_TICKERS
from src.features.indicators import MAX_WARMUP_TRADING_DAYS, compute_technical_features
from src.features.schema import FeatureRow, SplitBounds

LOOKBACK_TRADING_DAYS = 60  # progress/phase_f_lstm_decisions.md decision #1

# Chronological 3-way split ratios (row-count based). Exact ratios are left
# as an implementation detail by the decisions doc (#4) -- 70/15/15 chosen
# here to keep as much data as possible in `train` given the flux-score
# join bottleneck (only ~502 usable trading days per ticker, see README),
# while still giving `val` and `test` enough rows (~75 each) to be
# meaningful. See features/README.md for the full reasoning.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC is the remainder, not stored as a constant to avoid float drift.


def _load_full_price_history(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT date, adj_close FROM daily_prices WHERE ticker = ? ORDER BY date ASC",
        conn,
        params=(ticker,),
    )
    if df.empty:
        raise ValueError(f"No daily_prices rows found for ticker={ticker!r}")
    return df


def _load_flux_series(conn: sqlite3.Connection, ticker: str) -> tuple[list[str], list[float]]:
    """Full calendar-day flux_score series for a ticker, sorted ascending by date."""
    df = pd.read_sql_query(
        "SELECT date, flux_score FROM flux_scores_daily WHERE ticker = ? ORDER BY date ASC",
        conn,
        params=(ticker,),
    )
    if df.empty:
        raise ValueError(f"No flux_scores_daily rows found for ticker={ticker!r}")
    return df["date"].tolist(), df["flux_score"].tolist()


@dataclass
class JoinDiagnostics:
    ticker: str
    n_trading_days_total_history: int
    n_trading_days_in_flux_window: int
    n_exact_flux_matches: int
    n_forward_filled_flux_matches: int
    n_unmatched_dropped: int  # trading days in the flux window with NO usable flux_score at all
    warmup_nan_rows_dropped: int  # rows dropped for insufficient technical-indicator warmup


def _join_flux_score(
    price_df: pd.DataFrame, flux_dates: list[str], flux_scores: list[float], ticker: str
) -> tuple[pd.DataFrame, JoinDiagnostics]:
    """
    For each trading day in `price_df`, attach the flux_score for that exact
    calendar date if present in `flux_scores_daily`. If (and only if) no
    exact match exists, forward-fill from the most recent PRIOR calendar
    date's flux_score (never a later date -- that would be look-ahead).
    Trading days with no exact match AND no prior flux_score at all
    (i.e. before flux_scores_daily's own history starts) are dropped, not
    fabricated.

    Verified against the live DB (2026-07-22): every one of the 502 trading
    days per ticker inside the flux_scores_daily calendar window
    (2024-07-19..2026-07-21) has an EXACT date match, so the forward-fill
    branch is not exercised on the real dataset today -- it is kept for
    correctness/defensiveness (e.g. a future GDELT gap on a single calendar
    day) and is logged via JoinDiagnostics if it ever fires.
    """
    exact_map = dict(zip(flux_dates, flux_scores))
    min_flux_date = flux_dates[0]

    out_scores: list[Optional[float]] = []
    out_source_dates: list[Optional[str]] = []
    n_exact = 0
    n_ffill = 0
    n_unmatched = 0

    for d in price_df["date"]:
        if d in exact_map:
            out_scores.append(exact_map[d])
            out_source_dates.append(d)
            n_exact += 1
        elif d < min_flux_date:
            # Trading day predates flux_scores_daily's own history entirely --
            # nothing to forward-fill from. Drop, don't fabricate.
            out_scores.append(None)
            out_source_dates.append(None)
        else:
            # No exact match but flux history exists earlier -- forward-fill
            # from the most recent PRIOR calendar date (bisect keeps this
            # strictly backward-looking).
            idx = bisect.bisect_left(flux_dates, d) - 1
            if idx < 0:
                out_scores.append(None)
                out_source_dates.append(None)
            else:
                out_scores.append(flux_scores[idx])
                out_source_dates.append(flux_dates[idx])
                n_ffill += 1

    price_df = price_df.copy()
    price_df["flux_score"] = out_scores
    price_df["flux_score_source_date"] = out_source_dates

    in_window_mask = price_df["date"] >= min_flux_date
    n_in_window = int(in_window_mask.sum())
    # Only count "unmatched" WITHIN the flux-score-covered window -- rows
    # before flux_scores_daily's own history starts are expected to be NaN
    # here (they're only ever used for technical-indicator warmup, never
    # emitted in the final output) and are not a join failure.
    n_unmatched = int(price_df.loc[in_window_mask, "flux_score"].isna().sum())

    diag = JoinDiagnostics(
        ticker=ticker,
        n_trading_days_total_history=len(price_df),
        n_trading_days_in_flux_window=n_in_window,
        n_exact_flux_matches=n_exact,
        n_forward_filled_flux_matches=n_ffill,
        n_unmatched_dropped=n_unmatched,
        warmup_nan_rows_dropped=0,  # filled in by caller after indicator computation
    )
    return price_df, diag


def _assign_splits(dates: list[str]) -> tuple[list[str], dict[str, tuple[str, str]]]:
    """
    Chronological, contiguous, row-count-based 3-way split over `dates`
    (already sorted ascending). Returns (per-row split labels, boundaries).

    Why a window's features are allowed to reach backward across a split
    boundary but a window's LABEL must not: a supervised window for label
    date L uses feature days [L - LOOKBACK_TRADING_DAYS .. L - 1] (all
    strictly before L) and predicts something dated >= L. Because feature
    days are, by construction, always chronologically earlier than L, if L
    is assigned to `train` (L < val_start) then EVERY feature day of that
    window is also < val_start automatically -- a train window can never see
    val/test-period data. A `val` window's label is in [val_start, val_end),
    but its feature days may legitimately dip into `train`-period dates --
    that is not leakage (it's real, already-realized history, not future
    information), it's the normal and expected behavior of a sliding window
    evaluated near a chronological split boundary.
    """
    n = len(dates)
    train_end = round(n * TRAIN_FRAC)
    val_end = train_end + round(n * VAL_FRAC)
    val_end = min(val_end, n)
    train_end = min(train_end, val_end)

    labels = (["train"] * train_end) + (["val"] * (val_end - train_end)) + (["test"] * (n - val_end))
    boundaries = {
        "train": (dates[0], dates[train_end - 1]) if train_end > 0 else (None, None),
        "val": (dates[train_end], dates[val_end - 1]) if val_end > train_end else (None, None),
        "test": (dates[val_end], dates[-1]) if val_end < n else (None, None),
    }
    return labels, boundaries


def build_ticker_features(
    conn: sqlite3.Connection, ticker: str
) -> tuple[list[FeatureRow], list[SplitBounds], JoinDiagnostics]:
    """
    Full pipeline for one ticker: load full price history -> compute causal
    technical indicators over the FULL history (so warmup NaNs land in the
    pre-flux-window past, not in the output) -> join flux_score -> restrict
    to rows with a usable flux_score -> assign chronological splits ->
    validate every row.
    """
    price_df = _load_full_price_history(conn, ticker)
    price_df = compute_technical_features(price_df)

    flux_dates, flux_scores = _load_flux_series(conn, ticker)
    joined, diag = _join_flux_score(price_df, flux_dates, flux_scores, ticker)

    # Restrict to rows that have a usable flux_score (i.e. are inside or
    # forward-fillable from flux_scores_daily's own history). This is the
    # real bottleneck on this dataset: flux_scores_daily only goes back to
    # 2024-07-19, far short of most tickers' full price history.
    before_flux_restrict = len(joined)
    joined = joined[joined["flux_score"].notna()].reset_index(drop=True)

    # Also drop rows where the technical indicators haven't warmed up yet.
    # On the real data this should be zero, since flux_scores_daily starts
    # in mid-2024 and every ticker has >= 50 trading days of price history
    # before that -- but we check explicitly rather than assume.
    n_before_warmup_drop = len(joined)
    indicator_cols = ["daily_return", "sma_short", "sma_long", "volatility_10"]
    joined = joined[joined[indicator_cols].notna().all(axis=1)].reset_index(drop=True)
    warmup_dropped = n_before_warmup_drop - len(joined)
    diag.warmup_nan_rows_dropped = warmup_dropped

    if joined.empty:
        raise ValueError(f"No usable feature rows survived for ticker={ticker!r}")

    dates = joined["date"].tolist()
    split_labels, boundaries = _assign_splits(dates)
    joined["split"] = split_labels

    computed_at = datetime.now(timezone.utc).isoformat()
    rows: list[FeatureRow] = []
    for _, r in joined.iterrows():
        row = FeatureRow(
            ticker=ticker,
            date=r["date"],
            adj_close=float(r["adj_close"]),
            daily_return=float(r["daily_return"]),
            sma_short=float(r["sma_short"]),
            sma_long=float(r["sma_long"]),
            volatility_10=float(r["volatility_10"]),
            flux_score=float(r["flux_score"]),
            flux_score_source_date=str(r["flux_score_source_date"]),
            split=str(r["split"]),
            computed_at=computed_at,
        )
        row.validate()
        rows.append(row)

    split_bounds: list[SplitBounds] = []
    for split_name in ("train", "val", "test"):
        split_dates = [row.date for row in rows if row.split == split_name]
        n_rows = len(split_dates)
        start, end = boundaries.get(split_name, (None, None))
        # A date can serve as a WINDOW LABEL only if there are >= LOOKBACK_TRADING_DAYS
        # trading days of feature history strictly before it, counting across
        # split boundaries (see docstring of _assign_splits for why that's safe).
        usable_labels = [
            d for d in split_dates
            if dates.index(d) >= LOOKBACK_TRADING_DAYS
        ]
        split_bounds.append(SplitBounds(
            ticker=ticker,
            split=split_name,
            start_date=start or "",
            end_date=end or "",
            n_rows=n_rows,
            lookback_days=LOOKBACK_TRADING_DAYS,
            first_usable_label_date=usable_labels[0] if usable_labels else None,
            n_usable_labels=len(usable_labels),
            computed_at=computed_at,
        ))

    return rows, split_bounds, diag


def build_all(conn: sqlite3.Connection, tickers: list[str] = TRACKED_TICKERS):
    """Build feature rows + split bounds for every tracked ticker, isolating
    per-ticker failures (mirrors price/run_ingest.py's per-ticker error
    isolation) rather than letting one bad ticker abort the whole run."""
    results = {}
    errors = {}
    for ticker in tickers:
        try:
            rows, bounds, diag = build_ticker_features(conn, ticker)
            results[ticker] = (rows, bounds, diag)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, isolates per-ticker
            errors[ticker] = exc
    return results, errors
