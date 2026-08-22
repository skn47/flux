"""
From-scratch per-column scalers (no sklearn dependency in this repo's venv --
see requirements.txt; every other numeric package here is either
pandas/numpy, already installed, or a from-scratch model, so a ~15-line
scaler class is preferred over adding a new dependency for one class).

Two scalers, same dataclass shape (fit/transform/inverse_transform_col/
to_dict/from_dict):
  - `MinMaxScaler`: per-column min-max to [0, 1]. Used by the original
    price-level LSTM design; kept in place as a general-purpose class, no
    longer used by `lstm/dataset.py` as of R1 (see that module's docstring).
  - `StandardScaler`: per-column z-score. Used by `lstm/dataset.py` as of
    R1, for the new return/ratio-based feature set.

Both structurally guarantee the "fit on train partition only" rule from
`progress/phase_f_lstm_decisions.md` decision #3: `fit()` must be called with a
caller-supplied array, and this module never reaches into the DB or a full
dataset itself -- callers (see `lstm/dataset.py::build_ticker_dataset`)
always pass only the rows where `split == "train"`. `transform()` never
refits, so calling it on val/test data cannot leak val/test statistics into
the scaler by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MinMaxScaler:
    """Per-column Min-Max scaler to [0, 1]. Column order is the caller's
    responsibility (must match `feature_cols` order used at fit time)."""

    min_: np.ndarray = field(default=None)
    max_: np.ndarray = field(default=None)
    fitted: bool = False

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """X: (n_rows, n_cols), TRAIN-ONLY rows. Raises if called twice on
        purpose is not enforced (re-fitting is allowed structurally, since
        callers may legitimately re-fit per ticker/variant), but callers in
        this package only ever call fit() once per (ticker, variant) with a
        train-only array -- see dataset.py."""
        X = np.asarray(X, dtype=np.float64)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("MinMaxScaler.transform() called before fit()")
        X = np.asarray(X, dtype=np.float64)
        rng = self.max_ - self.min_
        rng = np.where(rng == 0, 1.0, rng)  # guard: constant train column -> no-op scale
        return (X - self.min_) / rng

    def inverse_transform_col(self, values: np.ndarray, col_idx: int) -> np.ndarray:
        """Inverse-transform a single column (used for the target column,
        which is always `adj_close`, col_idx 0 in both feature variants)."""
        if not self.fitted:
            raise RuntimeError("MinMaxScaler.inverse_transform_col() called before fit()")
        rng = self.max_[col_idx] - self.min_[col_idx]
        rng = rng if rng != 0 else 1.0
        return values * rng + self.min_[col_idx]

    def to_dict(self) -> dict:
        return {"min_": self.min_.tolist(), "max_": self.max_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "MinMaxScaler":
        return cls(min_=np.array(d["min_"], dtype=np.float64),
                    max_=np.array(d["max_"], dtype=np.float64), fitted=True)


@dataclass
class StandardScaler:
    """Per-column z-score (standard) scaler: (x - mean) / std. Same shape/
    discipline as `MinMaxScaler` above (fit/transform/inverse_transform_col/
    to_dict/from_dict, column order is the caller's responsibility, structurally
    fit-on-train-only by how callers in this package use it -- see
    `lstm/dataset.py::build_ticker_dataset`, the only call site).

    Added for R1 (see `lstm/dataset.py`'s module docstring): the new
    return/ratio-based feature set is centered near 0 with a comparatively
    small, roughly stationary spread, which z-scoring suits better than
    Min-Max (Min-Max would anchor a [0,1] range to the train sample's exact
    min/max, which is exactly the kind of fixed-range anchoring R1 is trying
    to get away from for the *scaling* step too, not just the feature
    choice)."""

    mean_: np.ndarray = field(default=None)
    std_: np.ndarray = field(default=None)
    fitted: bool = False

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """X: (n_rows, n_cols), TRAIN-ONLY rows. See MinMaxScaler.fit()'s
        docstring for the same re-fit-is-structurally-allowed-but-never-done
        note; callers in this package call this exactly once per
        (ticker, variant) with a train-only array."""
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("StandardScaler.transform() called before fit()")
        X = np.asarray(X, dtype=np.float64)
        std = np.where(self.std_ == 0, 1.0, self.std_)  # guard: constant train column -> no-op scale
        return (X - self.mean_) / std

    def inverse_transform_col(self, values: np.ndarray, col_idx: int) -> np.ndarray:
        """Inverse-transform a single column (used for the target column,
        `daily_return`, always at col_idx 0 -- see lstm/dataset.py)."""
        if not self.fitted:
            raise RuntimeError("StandardScaler.inverse_transform_col() called before fit()")
        std = self.std_[col_idx] if self.std_[col_idx] != 0 else 1.0
        return values * std + self.mean_[col_idx]

    def to_dict(self) -> dict:
        return {"mean_": self.mean_.tolist(), "std_": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScaler":
        return cls(mean_=np.array(d["mean_"], dtype=np.float64),
                    std_=np.array(d["std_"], dtype=np.float64), fitted=True)
