"""
Conformal calibration (CQR -- Conformalized Quantile Regression) for the
seq2seq model's MC-Dropout uncertainty bands. See lstm/seq2seq_README.md's
"Conformal calibration (Direction 4)" section for the empirical motivation:
the raw MC-Dropout 90%-nominal band achieves only ~5% empirical coverage
(backtest/seq2seq_backtest_summary.json) -- this module turns that into an
empirically-calibrated band using genuinely out-of-fold nonconformity scores
(see backtest/seq_walk_forward.py, lstm/run_conformal_calibration.py for how
the calibration set is built).

Method: standard two-sided CQR (Romano, Patterson & Candes 2019), the
SYMMETRIC single-q_hat variant, not the asymmetric split-tail form -- a
deliberate scope choice, stated explicitly (same "state scope reductions,
don't silently cut" discipline as backtest/walk_forward.py's module
docstring). Calibrated PER HORIZON STEP (steps 1..horizon each get their own
q_hat), not jointly across the whole 30-day path -- this guarantees marginal
per-step coverage, not simultaneous whole-path coverage; a further
refinement, not attempted here.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.backtest.seq_predictions import SeqFullPathPrediction


@dataclass
class ConformalCalibration:
    variant: str
    horizon: int
    alpha: float
    lo_pct: float
    hi_pct: float
    q_hat: list[float]            # length horizon, q_hat[k] is step (k+1)'s correction
    n_cal_per_step: list[int]      # length horizon
    fold_ids_used: list[int]
    computed_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ConformalCalibration":
        return ConformalCalibration(**d)


def cqr_nonconformity_scores(
    predictions: list[SeqFullPathPrediction],
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
) -> dict[int, list[float]]:
    """Per horizon step k=1..horizon: s_i = max(lo_k - actual_k, actual_k - hi_k)
    for each prediction, where [lo_k, hi_k] is that window's OWN raw
    MC-Dropout [lo_pct, hi_pct] percentile band at step k (same convention
    backtest/seq_calibration.py::compute_coverage_table uses, so scores
    computed here are directly comparable to that module's raw coverage
    numbers). A NEGATIVE score means the raw band already covered the actual
    value at that step; POSITIVE means it missed by that margin."""
    if not predictions:
        return {}
    horizon = predictions[0].actual_path.shape[0]
    scores: dict[int, list[float]] = {k: [] for k in range(1, horizon + 1)}
    for p in predictions:
        for k in range(horizon):
            samples_k = p.sample_paths[:, k]
            lo = np.percentile(samples_k, lo_pct)
            hi = np.percentile(samples_k, hi_pct)
            actual_k = p.actual_path[k]
            scores[k + 1].append(float(max(lo - actual_k, actual_k - hi)))
    return scores


def fit_cqr_calibration(
    scores_by_step: dict[int, list[float]],
    alpha: float = 0.10,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
    variant: str = "",
    fold_ids_used: Optional[list[int]] = None,
) -> ConformalCalibration:
    """Finite-sample-corrected split-conformal quantile per step (Vovk et
    al.): q_hat_k is the ceil((n+1)*(1-alpha))-th smallest of that step's n
    nonconformity scores. If n is too small for a valid order statistic
    (ceil((n+1)*(1-alpha)) > n), q_hat_k is set to +inf and that step's
    n_cal_per_step entry stays low, flagging it as under-powered -- NOT
    silently clipped to the max observed score, which would understate the
    true correction needed."""
    if not scores_by_step:
        raise ValueError("scores_by_step is empty -- no calibration data")
    horizon = len(scores_by_step)
    q_hat: list[float] = []
    n_cal: list[int] = []
    for k in range(1, horizon + 1):
        s = sorted(scores_by_step[k])
        n = len(s)
        n_cal.append(n)
        level = math.ceil((n + 1) * (1 - alpha))
        if n == 0 or level > n:
            q_hat.append(float("inf"))
        else:
            q_hat.append(s[level - 1])  # level is 1-indexed order statistic
    return ConformalCalibration(
        variant=variant,
        horizon=horizon,
        alpha=alpha,
        lo_pct=lo_pct,
        hi_pct=hi_pct,
        q_hat=q_hat,
        n_cal_per_step=n_cal,
        fold_ids_used=list(fold_ids_used or []),
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def apply_calibration(
    lo: np.ndarray,
    hi: np.ndarray,
    calibration: ConformalCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    """lo/hi: (..., horizon) raw MC-Dropout percentile bounds. Returns
    (lo_cal, hi_cal) of the same shape, widened per step by q_hat:
    lo_cal = lo - q_hat, hi_cal = hi + q_hat."""
    q = np.asarray(calibration.q_hat, dtype=np.float64)
    if lo.shape[-1] != q.shape[0]:
        raise ValueError(f"lo/hi last-dim {lo.shape[-1]} != calibration horizon {q.shape[0]}")
    return lo - q, hi + q


def save_calibration(calibration: ConformalCalibration, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(calibration.to_dict(), f, indent=2)


def load_calibration(path: Path | str) -> ConformalCalibration:
    with open(path) as f:
        return ConformalCalibration.from_dict(json.load(f))
