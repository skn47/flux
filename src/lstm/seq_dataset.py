"""
30-day-forward windowing for the seq2seq price-reaction model (see
lstm/seq2seq_model.py). Sibling to lstm/dataset.py, NOT a modification of it
-- lstm/dataset.py's 1-day-ahead `Window`/`build_ticker_dataset` remain
byte-identical and still back the production `lstm/model.py::PriceLSTM`
pipeline. See lstm/seq2seq_README.md for the full rationale for building
this alongside rather than in place.

Reuses, unmodified, from lstm/dataset.py: `load_ticker_frame` (features_daily
read + the two in-memory R1 ratio columns), `TICKER2ID`, `TARGET_COL`,
`VARIANTS`/`UNSCALED_FEATURE_COLS`, `split_for_date`, `_scaled_cols`, and
from src.lstm/scaling.py: `StandardScaler` (same train-only-fit discipline, same
`flux_score` left-unscaled exception).

The one load-bearing change from lstm/dataset.py's window loop
-------------------------------------------------------------------
lstm/dataset.py's `build_ticker_dataset` labels window `i` with row `i`
itself (the very next day, `y_scaled = scaled_all[i, target_col_idx]`) --
confirmed directly in that file, this is the exact place "predict 1 day
ahead" is hardcoded. Here, a window's target is instead the FULL
`daily_return` path over `[i, i+horizon)`:

    for i in range(lookback, n - horizon):
        X = scaled_all[i - lookback : i, :]              # unchanged
        y_scaled = scaled_all[i : i + horizon, target_col_idx]   # (horizon,)
        label_dates = dates[i : i + horizon]

`range(lookback, n - horizon)` (not `range(lookback, n)`) is the forward-
availability guard -- the last `horizon - 1` rows of a ticker's history can
no longer serve as a window's start, since their full label path would run
off the end of the available data.

Split assignment: a window's split is classified from `label_dates[0]`
(the FIRST day of its target path, i.e. `dates[i]`) -- the same date
lstm/dataset.py's `Window` uses for its single `label_date`. But
`label_dates[0]` being in-fold does NOT guarantee `label_dates[-1]` (up to
`horizon - 1` trading days later, where the target actually REALIZES) is
too -- a leakage class a 1-day target could never hit. `build_ticker_seq_dataset`
therefore EXCLUDES (not merely mislabels) any window whose `label_dates[-1]`
would reach into the next split's date range -- see the `val_start_date`/
`test_start_date` exclusion checks in the window loop below. This is the
actual leakage-PREVENTION step; `verify_no_leakage_seq` below is the
independent check that this prevention worked, not a substitute for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import sqlite3

from config.mvp_scope import TRACKED_TICKERS
from src.flux_engine.projection import FluxTrajectoryProjector
from src.lstm.dataset import (
    DEFAULT_DB_PATH,
    LOOKBACK,
    TARGET_COL,
    TICKER2ID,
    UNSCALED_FEATURE_COLS,
    VARIANTS,
    _scaled_cols,
    load_ticker_frame,
    split_for_date,
)
from src.lstm.scaling import StandardScaler

HORIZON = 30  # trading days ahead -- see lstm/seq2seq_README.md


@dataclass
class SeqWindow:
    ticker: str
    ticker_id: int
    feature_start_date: str
    feature_end_date: str        # t = last day the encoder sees; also the flux projection's as_of_date
    label_dates: list[str]        # [t+1 .. t+horizon], real trading-calendar dates, ascending
    X: np.ndarray                  # (LOOKBACK, n_features), scaled -- identical construction to lstm.dataset.Window.X
    y_scaled: np.ndarray            # (horizon,) scaled daily_return path
    y_raw: np.ndarray               # (horizon,) raw daily_return path -- ground truth for metrics
    flux_traj: np.ndarray           # (horizon,), UNSCALED, leakage-safe projected flux_score path
    split: str                      # "train" | "val" | "test", classified from label_dates[0] only


def build_ticker_seq_dataset(
    ticker: str,
    feature_cols: list[str],
    projector: FluxTrajectoryProjector,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
    db_path: Path | str = DEFAULT_DB_PATH,
    split_boundaries: Optional[dict] = None,
) -> tuple[list[SeqWindow], StandardScaler]:
    """Builds every usable 30-day-forward window for one ticker/variant.
    Mirrors lstm.dataset.build_ticker_dataset's scaling/split-assignment
    logic exactly (same train-only StandardScaler.fit() call, same
    split_boundaries None-means-"use stored features_daily.split" contract
    for R5/walk-forward compatibility) -- only the window-slicing loop at the
    bottom differs, per this module's docstring."""
    df = load_ticker_frame(ticker, db_path=db_path)

    scaled_cols = _scaled_cols(feature_cols)

    if split_boundaries is None:
        train_mask = (df["split"] == "train").to_numpy()
        splits_col: list[Optional[str]] = df["split"].tolist()
    else:
        missing = [k for k in ("train", "val", "test") if k not in split_boundaries]
        if missing:
            raise ValueError(f"split_boundaries missing required key(s): {missing}")
        splits_col = [split_for_date(d, split_boundaries) for d in df["date"]]
        train_mask = np.array([s == "train" for s in splits_col], dtype=bool)

    if not train_mask.any():
        raise ValueError(f"{ticker}: no train rows found")
    train_only = df.loc[train_mask, scaled_cols].to_numpy(dtype=np.float64)

    scaler = StandardScaler().fit(train_only)  # the ONLY fit() call, train rows only
    scaled_part = scaler.transform(df[scaled_cols].to_numpy(dtype=np.float64))

    scaled_all = np.empty((len(df), len(feature_cols)), dtype=np.float64)
    for j, col in enumerate(feature_cols):
        if col in UNSCALED_FEATURE_COLS:
            scaled_all[:, j] = df[col].to_numpy(dtype=np.float64)
        else:
            scaled_all[:, j] = scaled_part[:, scaled_cols.index(col)]

    target_col_idx = feature_cols.index(TARGET_COL)
    raw_target = df[TARGET_COL].to_numpy(dtype=np.float64)
    dates = df["date"].tolist()

    # Forward-realization boundary dates, derived from splits_col itself (the
    # per-row split labels already computed above, whichever source they
    # came from) rather than a second DB query -- the first date labeled
    # "val"/"test" in THIS ticker's own row sequence. A window's target
    # realizes `horizon` days after its label starts; if that realization
    # date reaches the NEXT split's start, the window must be EXCLUDED
    # entirely (not merely mislabeled) -- classifying a window by
    # label_dates[0] alone is not sufficient to prevent this, since
    # label_dates[0] being in-fold does not guarantee label_dates[-1] (up to
    # horizon-1 trading days later) is too. This is the actual leakage-
    # PREVENTION step; lstm/seq_dataset.py::verify_no_leakage_seq below is
    # the independent check that this prevention worked, not a substitute
    # for it.
    val_start_date = next((d for d, s in zip(dates, splits_col) if s == "val"), None)
    test_start_date = next((d for d, s in zip(dates, splits_col) if s == "test"), None)

    windows: list[SeqWindow] = []
    n = len(df)
    for i in range(lookback, n - horizon):
        split_i = splits_col[i]
        if split_i is None:
            # R5-style: this row's date falls outside every split_boundaries
            # range (e.g. after a fold's own test-end date) -- not visible to
            # this fold, no window built. Same convention as lstm/dataset.py.
            continue
        label_dates = dates[i: i + horizon]
        label_end = label_dates[-1]
        if split_i == "train" and val_start_date is not None and label_end >= val_start_date:
            continue  # target would realize into val -- exclude, don't just mislabel
        if split_i == "val" and test_start_date is not None and label_end >= test_start_date:
            continue  # target would realize into test -- exclude, don't just mislabel
        X = scaled_all[i - lookback: i, :].astype(np.float32)
        as_of = date.fromisoformat(dates[i - 1])
        target_dates = [date.fromisoformat(d) for d in label_dates]
        flux_traj = projector.project(ticker, as_of, target_dates)
        windows.append(
            SeqWindow(
                ticker=ticker,
                ticker_id=TICKER2ID[ticker],
                feature_start_date=dates[i - lookback],
                feature_end_date=dates[i - 1],
                label_dates=label_dates,
                X=X,
                y_scaled=scaled_all[i: i + horizon, target_col_idx].astype(np.float32),
                y_raw=raw_target[i: i + horizon].astype(np.float32),
                flux_traj=flux_traj.astype(np.float32),
                split=split_i,
            )
        )
    return windows, scaler


def build_pooled_seq_dataset(
    variant: str,
    tickers: list[str] = TRACKED_TICKERS,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
    db_path: Path | str = DEFAULT_DB_PATH,
    split_boundaries: Optional[dict] = None,
    projector: Optional[FluxTrajectoryProjector] = None,
) -> tuple[dict[str, list[SeqWindow]], dict[str, StandardScaler]]:
    """Pools windows across all tickers (same shared-model rationale as
    lstm.dataset.build_pooled_dataset). Builds ONE FluxTrajectoryProjector
    (a single global cluster pass) and reuses it across every ticker/window
    unless the caller supplies one, since `flux_conditioning="none"` callers
    still pay for the projector's construction otherwise -- callers that
    know they don't need flux_traj at all can still call this (the
    projection is always computed, since a "none" model simply ignores the
    column; keeping one code path is preferred over threading a
    skip-projection flag through for a case that isn't the bottleneck)."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}, expected one of {list(VARIANTS)}")
    feature_cols = VARIANTS[variant]
    projector = projector or FluxTrajectoryProjector(db_path=db_path)

    by_split: dict[str, list[SeqWindow]] = {"train": [], "val": [], "test": []}
    scalers: dict[str, StandardScaler] = {}
    for ticker in tickers:
        windows, scaler = build_ticker_seq_dataset(
            ticker, feature_cols, projector, lookback=lookback, horizon=horizon,
            db_path=db_path, split_boundaries=split_boundaries,
        )
        scalers[ticker] = scaler
        for w in windows:
            by_split[w.split].append(w)
    return by_split, scalers


def verify_no_leakage_seq(
    by_split: dict[str, list[SeqWindow]],
    db_path: Path | str = DEFAULT_DB_PATH,
    fold_id: Optional[int] = None,
) -> dict:
    """
    Independent structural leakage check for SeqWindows -- generalizes
    lstm.dataset.verify_no_leakage to a multi-day-forward target. Reads the
    true train/val/test date boundaries directly from the DB (not any
    in-memory assumption from dataset construction), then checks, per
    window:

      1. (same as the 1-day check) label_dates[0] falls within its own
         split's date range; for TRAIN windows, feature_end_date is strictly
         before the val split's start date (a train window's INPUT context
         never reaches into val/test).

      2. (NEW -- required because the target now realizes up to `horizon`
         days after label_dates[0], a leakage class a 1-day target could
         never hit, since its realization date and label date were the same
         day): a TRAIN window's label_dates[-1] (its target's REALIZATION
         date) must ALSO fall strictly before the val split's start date; a
         VAL window's label_dates[-1] must fall strictly before the test
         split's start date. `label_dates[0]` being in-fold does not
         guarantee `label_dates[-1]` (up to `horizon - 1` trading days
         later) is too.

    Raises AssertionError on the first violation found.
    """
    if fold_id is None:
        conn = sqlite3.connect(str(db_path))
        bounds = pd.read_sql_query(
            "SELECT ticker, split, start_date, end_date FROM feature_split_bounds", conn
        )
        conn.close()
    else:
        from src.lstm.walk_forward_bounds import read_bounds_frame
        bounds = read_bounds_frame(fold_id, db_path=db_path)

    bounds_by_ticker: dict[str, dict[str, tuple[str, str]]] = {}
    for _, row in bounds.iterrows():
        bounds_by_ticker.setdefault(row["ticker"], {})[row["split"]] = (row["start_date"], row["end_date"])

    n_checked = {"train": 0, "val": 0, "test": 0}
    for split_name, windows in by_split.items():
        for w in windows:
            b = bounds_by_ticker[w.ticker]
            lo, hi = b[split_name]
            label_start = w.label_dates[0]
            label_end = w.label_dates[-1]
            assert lo <= label_start <= hi, (
                f"LEAKAGE: {w.ticker} {split_name} window label_dates[0]={label_start} "
                f"outside its own split's date range [{lo}, {hi}]"
            )
            if split_name == "train":
                val_start = b.get("val", (None, None))[0]
                if val_start is not None:
                    assert w.feature_end_date < val_start, (
                        f"LEAKAGE: {w.ticker} TRAIN window feature_end_date="
                        f"{w.feature_end_date} >= val_start={val_start}"
                    )
                    assert label_end < val_start, (
                        f"LEAKAGE: {w.ticker} TRAIN window's target REALIZES at "
                        f"label_dates[-1]={label_end} >= val_start={val_start} -- "
                        f"label_dates[0]={label_start} is in train but the 30-day-forward "
                        f"target crosses into val"
                    )
            if split_name == "val":
                test_start = b.get("test", (None, None))[0]
                if test_start is not None:
                    assert label_end < test_start, (
                        f"LEAKAGE: {w.ticker} VAL window's target REALIZES at "
                        f"label_dates[-1]={label_end} >= test_start={test_start} -- "
                        f"label_dates[0]={label_start} is in val but the 30-day-forward "
                        f"target crosses into test"
                    )
            n_checked[split_name] += 1
    return {"windows_checked": n_checked, "status": "no leakage found"}
