"""
Windowing / dataset construction for the LSTM price-reaction model.

Reads `features_daily` (built by `features/`, see `features/README.md`),
constructs causal, non-centered 60-trading-day sliding windows per ticker,
and assigns each window to train/val/test by its LABEL date's split -- never
by majority vote or by the split of its feature rows. This is the exact rule
documented in `features/README.md` ("why a window's features may cross a
split boundary but its label never can"):

    window with label date L uses feature rows [L-60 .. L-1] (60 trading
    days, all strictly before L) and label = daily_return at L. The window's
    split is whatever `features_daily.split` says for row L. A window may
    read feature rows dated earlier than L's split boundary (e.g. a val
    window's early feature rows can be dated in train) -- that is not
    leakage, only a train window reading INTO val/test would be, and cannot
    happen here because feature rows are always strictly before L by
    construction.

R1 REWORK (see `/home/h/.claude/plans/yes-kick-off-phase-stateful-trinket.md`,
"Status update (2026-07-23)"): this module previously predicted the raw
`adj_close` PRICE LEVEL, using raw `adj_close` as both an input feature and
the target. That was the actual root cause of the prior backtest's failure
(documented in `backtest/README.md`'s Phase G audit and `lstm/README.md`'s
"Known gaps"): all 5 tracked tickers rallied past the train-fitted scaler's
price range during the test window, so every test-period input/target was
fed to the model as a value structurally outside the [0,1]-ish range it was
ever trained on -- an absolute price level, scaled by a fixed train-only
range, is GUARANTEED to leave that range once a stock makes a new all-time
high, which any sustained rally will eventually do. This is a target/feature
framing bug, not a leakage bug (the train-only scaler fit itself was and
remains correct).

R1's fix: reframe both the input features and the target around
scale-relative, comparatively stationary quantities instead of an absolute
price level:

    BASELINE_FEATURE_COLS = [daily_return, price_to_sma_long, sma_ratio,
                              volatility_10]                        (4 cols)
    FLUX_FEATURE_COLS    = BASELINE_FEATURE_COLS + [flux_score]   (5 cols)
    TARGET_COL = daily_return   (always index 0 in both variants above)

`price_to_sma_long = adj_close / sma_long` and `sma_ratio = sma_short /
sma_long` are NEW columns, computed in-memory in `load_ticker_frame()` below
from `features_daily`'s already-stored `adj_close`/`sma_short`/`sma_long` --
this deliberately does NOT change the `features_daily` DB schema or
`features/build.py`; per `features/README.md`, that package defers all
scaling/derived-ratio work to the consuming package (originally written for
scaling, extended here to cover derived ratios for the same reason: they can
only be correctly computed once the actual consumer -- this module -- is
known, and doing it here keeps `features_daily` a stable, reusable raw
table).

Why this is expected to help (a HYPOTHESIS being tested by the R1 gate check
in `lstm/README.md`, not a guarantee): a daily return and a price-to-moving-
average ratio both stay in a comparatively narrow, roughly regime-independent
band even during a strong sustained trend -- e.g. `price_to_sma_long` for a
stock steadily grinding upward tends to hover in a bounded neighborhood of 1.0
(how far price has stretched above its own 50-day trend), rather than growing
without bound the way the raw price itself does. This is NOT a guarantee: an
unprecedented, accelerating trend (which is close to what the 2026 test
window actually was) can still push `price_to_sma_long` outside its train-fit
range -- just far less severely than raw price level did, since the ratio's
whole point is to net out the trend itself, not to be immune to regime shifts
in how far price can stretch from that trend. Whether this hypothesis holds
on the real backtest is exactly what the R1 gate check (see `lstm/README.md`)
reports, honestly, either way.

Scaling: a per-(ticker, variant) z-score `StandardScaler`
(`lstm/scaling.py`), REPLACING the old `MinMaxScaler`, fit ONLY on that
ticker's TRAIN-split rows, then applied (never re-fit) to that ticker's full
row range before windows are sliced out -- same train-only-fit discipline as
before (see `build_ticker_dataset()` below, the only place `fit()` is ever
called). `flux_score` is the one exception: it stays UNSCALED (it is
already bounded [0,1] by construction as a noisy-OR output -- see
`progress/flux_score_timeseries_findings.md` -- and z-scoring an
already-understood, already-saturated [0.85, 0.96]-mean signal would only
obscure that residual, not clarify it). See `UNSCALED_FEATURE_COLS` and the
module-level assertions below.

R5 REWORK (walk-forward validation, see `backtest/walk_forward.py` and
`/home/h/.claude/plans/yes-kick-off-phase-stateful-trinket.md`'s "R5"
section): `build_ticker_dataset()` and `build_pooled_dataset()` below gained
an optional `split_boundaries` parameter (default `None`). `None` means
"behave exactly as before" -- split labels come from the stored
`features_daily.split` column, unchanged, byte-identical to pre-R5 output
(regression-checked, see `lstm/README.md`'s R5 addendum). When a caller
supplies `split_boundaries` (a `{"train": (lo, hi), "val": (lo, hi), "test":
(lo, hi)}` dict of inclusive date-string ranges), each row's split is instead
computed IN-MEMORY from its own raw `date` column -- used by the walk-forward
fold loop to carve several different train/val/test date-range partitions
out of the SAME underlying `features_daily` rows, without ever touching the
DB's stored `split` column or `feature_split_bounds` (a purely in-memory
override, per the task's explicit requirement). A row whose `date` falls
outside all three ranges gets no window built from it at all (silently
excluded, not assigned a split) -- this is what makes an earlier fold's
dataset a genuine truncation of history at that fold's own test-end date,
not just a differently-labeled view of the complete 2024-07-19..2026-07-22
range.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.mvp_scope import TRACKED_TICKERS
from src.lstm.scaling import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "events.db"

LOOKBACK = 60  # trading days, per progress/phase_f_lstm_decisions.md decision #1

BASELINE_FEATURE_COLS: list[str] = [
    "daily_return", "price_to_sma_long", "sma_ratio", "volatility_10",
]
FLUX_FEATURE_COLS: list[str] = BASELINE_FEATURE_COLS + ["flux_score"]

TARGET_COL = "daily_return"  # always index 0 in both variants above

VARIANTS: dict[str, list[str]] = {
    "baseline": BASELINE_FEATURE_COLS,
    "flux": FLUX_FEATURE_COLS,
}

# Columns that must NOT be z-scored (see module docstring's "Scaling"
# section). `flux_score` is already bounded [0,1] by construction.
UNSCALED_FEATURE_COLS: frozenset[str] = frozenset({"flux_score"})

TICKER2ID: dict[str, int] = {t: i for i, t in enumerate(TRACKED_TICKERS)}


def _scaled_cols(feature_cols: list[str]) -> list[str]:
    """feature_cols with UNSCALED_FEATURE_COLS removed, order preserved."""
    return [c for c in feature_cols if c not in UNSCALED_FEATURE_COLS]


# R5: split_boundaries type, keyed "train"/"val"/"test" -> inclusive
# (start_date, end_date) date-string tuples over the raw `date` column. Kept
# as a plain dict (not a dataclass) to match how it's threaded straight
# through from backtest/walk_forward.py's fold-computation code, and stored
# straight into lstm/walk_forward_bounds.py's table with the same shape.
SplitBoundaries = dict  # dict[str, tuple[str, str]] -- see module docstring


def split_for_date(date: str, split_boundaries: dict) -> Optional[str]:
    """
    R5: classifies a single raw `date` string against a `split_boundaries`
    dict (see module docstring), checking train/val/test in that order.
    Returns the split name, or None if `date` falls outside all three
    ranges (e.g. a date after this fold's test end -- not yet "visible" to a
    more history-constrained earlier fold). Public (not `_`-prefixed)
    because `backtest/walk_forward.py` reuses this exact classification rule
    to label the full features_daily frame it feeds to the SMA-momentum
    control (backtest/cross_sectional.py::sma_momentum_value_frame) and the
    random-rank control's date list, so both signal paths and the model
    path draw split boundaries from one single implementation, not two
    independently-written ones that could silently drift apart.
    """
    for name in ("train", "val", "test"):
        lo, hi = split_boundaries[name]
        if lo <= date <= hi:
            return name
    return None


# --- Module-load-time verification (this project's "assert, don't just
# document" discipline -- see task spec / lstm/README.md's verification
# section): TARGET_COL must never be an unscaled column, and its position
# among the SCALED columns must match its position in feature_cols, since
# downstream code (this module's build_ticker_dataset, lstm/run_train.py,
# backtest/predictions.py) all compute `feature_cols.index(TARGET_COL)` and
# feed that index straight into the StandardScaler's `inverse_transform_col`,
# which only knows about the scaled-column subset's own ordering. Both are
# true by construction today (daily_return is always first, flux_score, if
# present, is always last) -- asserted here so a future column-order change
# fails loudly at import time instead of silently mis-inverse-transforming.
assert TARGET_COL not in UNSCALED_FEATURE_COLS, (
    f"TARGET_COL={TARGET_COL!r} must never be in UNSCALED_FEATURE_COLS"
)
for _variant_name, _variant_cols in VARIANTS.items():
    _sc = _scaled_cols(_variant_cols)
    assert _sc.index(TARGET_COL) == _variant_cols.index(TARGET_COL), (
        f"{_variant_name}: TARGET_COL's position among scaled columns "
        f"({_sc.index(TARGET_COL)}) must match its position in feature_cols "
        f"({_variant_cols.index(TARGET_COL)}) -- inverse_transform_col's "
        f"col_idx would silently target the wrong column otherwise."
    )
del _variant_name, _variant_cols, _sc


@dataclass
class Window:
    ticker: str
    ticker_id: int
    label_date: str
    feature_start_date: str
    feature_end_date: str
    X: np.ndarray       # (LOOKBACK, n_features), SCALED (train-only-fit scaler; flux_score, if present, left raw)
    y_scaled: float      # scaled daily_return at label_date
    y_raw: float          # raw (unscaled) daily_return at label_date -- ground truth for metrics (R1: was raw adj_close)
    split: str            # "train" | "val" | "test", from the LABEL row only


def load_ticker_frame(ticker: str, db_path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """All rows for `ticker` from features_daily, ascending by date, plus two
    NEW in-memory-only derived ratio columns (R1, see module docstring):
    `price_to_sma_long = adj_close / sma_long` and
    `sma_ratio = sma_short / sma_long`. Uses the FULL row range (not filtered
    to a single split) because val/test windows legitimately need feature
    context reaching back into train-dated rows."""
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        """
        SELECT date, adj_close, daily_return, sma_short, sma_long,
               volatility_10, flux_score, split
        FROM features_daily
        WHERE ticker = ?
        ORDER BY date ASC
        """,
        conn,
        params=(ticker,),
    )
    conn.close()
    if not df["date"].is_monotonic_increasing:
        raise ValueError(f"{ticker}: features_daily rows not sorted ascending by date")
    if df["date"].duplicated().any():
        raise ValueError(f"{ticker}: duplicate date rows in features_daily")

    # R1 derived features -- NOT persisted to features_daily (features/README.md's
    # "scaling/derived-ratio work deferred to the consumer" design). Computed
    # here from already-causal stored columns (sma_short/sma_long/adj_close
    # are all trailing-only per features/indicators.py), so these two new
    # columns are causal by construction too -- no new leakage surface.
    df["price_to_sma_long"] = df["adj_close"] / df["sma_long"]
    df["sma_ratio"] = df["sma_short"] / df["sma_long"]
    return df


def build_ticker_dataset(
    ticker: str,
    feature_cols: list[str],
    lookback: int = LOOKBACK,
    db_path: Path | str = DEFAULT_DB_PATH,
    split_boundaries: Optional[dict] = None,
) -> tuple[list[Window], StandardScaler]:
    """
    Builds every usable window for one ticker/variant, and the scaler used
    to produce them. The scaler's .fit() call below is given ONLY the rows
    classified as "train" -- this is the single place in the whole package
    a scaler is fit, and it structurally cannot see val/test rows.

    R1: scaler is now a `StandardScaler` (z-score), fit only on the
    non-`flux_score` columns; `flux_score` (if present in feature_cols) is
    copied through into the final scaled matrix raw/untouched (see module
    docstring's "Scaling" section).

    R5: `split_boundaries` (default `None`) -- see module docstring's "R5
    REWORK" section. `None` reproduces the exact original code path
    (`df["split"]` straight from the DB); given, each row's split is instead
    computed in-memory via `split_for_date()`, and rows outside every
    boundary are dropped from the returned windows list entirely (not
    included with any split label) -- this is what truncates a fold's
    dataset to only what that fold's own history cutoff could have seen.
    """
    df = load_ticker_frame(ticker, db_path=db_path)

    scaled_cols = _scaled_cols(feature_cols)

    if split_boundaries is None:
        # Unmodified original path -- byte-identical to pre-R5 behavior.
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

    scaler = StandardScaler().fit(train_only)  # <-- the ONLY fit() call, train rows only
    scaled_part = scaler.transform(df[scaled_cols].to_numpy(dtype=np.float64))

    # Assemble the full (n_rows, n_features) matrix in feature_cols order,
    # copying UNSCALED_FEATURE_COLS (flux_score) through raw/untouched.
    scaled_all = np.empty((len(df), len(feature_cols)), dtype=np.float64)
    for j, col in enumerate(feature_cols):
        if col in UNSCALED_FEATURE_COLS:
            scaled_all[:, j] = df[col].to_numpy(dtype=np.float64)
        else:
            scaled_all[:, j] = scaled_part[:, scaled_cols.index(col)]

    target_col_idx = feature_cols.index(TARGET_COL)
    raw_target = df[TARGET_COL].to_numpy(dtype=np.float64)  # raw daily_return (R1: was raw adj_close)
    dates = df["date"].tolist()

    windows: list[Window] = []
    n = len(df)
    for i in range(lookback, n):
        split_i = splits_col[i]
        if split_i is None:
            # R5 only: this row's date falls outside every boundary in
            # split_boundaries (e.g. after this fold's own test-end date) --
            # not yet "visible" to this fold, so no window is built from it.
            continue
        X = scaled_all[i - lookback: i, :].astype(np.float32)
        windows.append(
            Window(
                ticker=ticker,
                ticker_id=TICKER2ID[ticker],
                label_date=dates[i],
                feature_start_date=dates[i - lookback],
                feature_end_date=dates[i - 1],
                X=X,
                y_scaled=float(scaled_all[i, target_col_idx]),
                y_raw=float(raw_target[i]),
                split=split_i,
            )
        )
    return windows, scaler


def build_pooled_dataset(
    variant: str,
    tickers: list[str] = TRACKED_TICKERS,
    lookback: int = LOOKBACK,
    db_path: Path | str = DEFAULT_DB_PATH,
    split_boundaries: Optional[dict] = None,
) -> tuple[dict[str, list[Window]], dict[str, StandardScaler]]:
    """
    Pools windows across all tickers (shared-model design, see lstm/README.md
    "Per-ticker or shared model") into {"train": [...], "val": [...],
    "test": [...]}, plus each ticker's own fitted scaler (needed later to
    inverse-transform that ticker's predictions back to raw return units).

    R5: `split_boundaries`, if given, is passed straight through to every
    ticker's `build_ticker_dataset()` call UNCHANGED -- all tracked tickers
    share one trading calendar (confirmed directly against the live DB, see
    backtest/walk_forward.py), so one shared boundaries dict correctly
    applies to every ticker; `None` (default) reproduces pre-R5 behavior
    exactly.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}, expected one of {list(VARIANTS)}")
    feature_cols = VARIANTS[variant]

    by_split: dict[str, list[Window]] = {"train": [], "val": [], "test": []}
    scalers: dict[str, StandardScaler] = {}
    for ticker in tickers:
        windows, scaler = build_ticker_dataset(
            ticker, feature_cols, lookback=lookback, db_path=db_path,
            split_boundaries=split_boundaries,
        )
        scalers[ticker] = scaler
        for w in windows:
            by_split[w.split].append(w)
    return by_split, scalers


def verify_no_leakage(
    by_split: dict[str, list[Window]],
    db_path: Path | str = DEFAULT_DB_PATH,
    fold_id: Optional[int] = None,
) -> dict:
    """
    Independent structural leakage check (verification discipline required
    by the task): for every ticker, determine the true train/val/test date
    boundaries directly from the DB (not reusing any in-memory assumption
    from dataset construction), then confirm:
      1. every TRAIN window's label_date falls within that ticker's train
         date range, and its feature_end_date is < the val start date
         (i.e. a train window never reads into val/test).
      2. every VAL/TEST window's label_date falls within its own split's
         date range (feature dates are allowed to reach earlier, into train
         -- not checked as a violation, only label dates are).
    Raises AssertionError on the first violation found. Returns a small
    summary dict on success.

    R1: UNCHANGED from before (windowing/split-assignment/leakage-guard
    logic is not touched by the target/feature rework) -- only the Window's
    X/y_scaled/y_raw contents differ.

    R5: optional `fold_id`. `None` (default) reads bounds from
    `feature_split_bounds` exactly as before -- unchanged. Given, reads
    bounds from `lstm/walk_forward_bounds.py`'s `walk_forward_bounds` table
    `WHERE fold_id = ?` instead -- the fold-scoped, persisted analogue of
    `feature_split_bounds` (written once per fold by
    `backtest/walk_forward.py` before that fold's models are trained). Same
    assertion logic either way -- only the bounds SOURCE differs.
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
            assert lo <= w.label_date <= hi, (
                f"LEAKAGE: {w.ticker} {split_name} window label_date={w.label_date} "
                f"outside its own split's date range [{lo}, {hi}]"
            )
            if split_name == "train":
                # A train window's feature context must never reach into val/test.
                val_start = b.get("val", (None, None))[0]
                if val_start is not None:
                    assert w.feature_end_date < val_start, (
                        f"LEAKAGE: {w.ticker} TRAIN window feature_end_date="
                        f"{w.feature_end_date} >= val_start={val_start}"
                    )
            n_checked[split_name] += 1
    return {"windows_checked": n_checked, "status": "no leakage found"}
