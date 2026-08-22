"""
Per-horizon-step prediction-accuracy metrics for the seq2seq model, plus
multi-step, model-free persistence baselines -- the horizon-generalized
analogue of lstm/evaluate.py (reused directly per step here, not
reimplemented).

Per-horizon-step, not a single pooled number, because error should grow with
horizon distance if the model (and the task) behave sensibly -- collapsing
all 30 steps into one MAE would hide that shape. `compute_metrics_per_step`
returns one lstm.evaluate.compute_metrics()-shaped dict per step
k=0..horizon-1 (each independently re-verified via
lstm.evaluate.assert_metrics_agree, same discipline as lstm/run_train.py),
plus one pooled "all_steps" dict (all n_windows*horizon points combined) for
a single headline number.

Multi-step persistence baselines, generalizing lstm/evaluate.py's two 1-day
baselines:
  - "zero"  -- predict every one of the horizon days' returns = 0.
  - "prior" -- predict every one of the horizon days' returns = the single
    last REALIZED return the model actually saw (feature_end_date's
    daily_return), held constant across the whole forecast window. This is
    the natural multi-step generalization of lstm/evaluate.py's "tomorrow's
    return = today's return" rule: there is no realized data yet for steps
    2..horizon, so the only model-free constant available to every step is
    the one already-realized value at feature_end_date.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.lstm.dataset import DEFAULT_DB_PATH, load_ticker_frame
from src.lstm.evaluate import assert_metrics_agree, compute_metrics, recompute_metrics_independently
from src.lstm.seq_dataset import SeqWindow


def compute_metrics_per_step(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true, y_pred: (n_windows, horizon), raw units. Returns
    {"per_step": [metrics_dict, ...] (len horizon), "all_steps": metrics_dict}
    -- every per-step and the pooled metric independently re-verified before
    being returned, same pattern as lstm/run_train.py::evaluate_split."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    horizon = y_true.shape[1]

    per_step = []
    for k in range(horizon):
        m_vec = compute_metrics(y_true[:, k], y_pred[:, k])
        m_indep = recompute_metrics_independently(y_true[:, k].tolist(), y_pred[:, k].tolist())
        assert_metrics_agree(m_vec, m_indep)
        per_step.append(m_vec)

    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    m_all_vec = compute_metrics(flat_true, flat_pred)
    m_all_indep = recompute_metrics_independently(flat_true.tolist(), flat_pred.tolist())
    assert_metrics_agree(m_all_vec, m_all_indep)

    return {"per_step": per_step, "all_steps": m_all_vec}


def compute_persistence_baseline_seq(
    windows: list[SeqWindow], db_path: Path | str = DEFAULT_DB_PATH
) -> dict:
    """Multi-step, model-free baselines on raw daily_return units -- see
    module docstring. `y_prior` is fetched fresh per ticker via
    lstm.dataset.load_ticker_frame, NOT reconstructed from any window's
    scaled feature matrix -- zero dependency on scaler/model code, same
    discipline as lstm/run_train.py's 1-day persistence baseline."""
    raw_return_by_ticker_date: dict[str, dict[str, float]] = {}
    y_true = []
    y_prior_const = []
    for w in windows:
        if w.ticker not in raw_return_by_ticker_date:
            frame = load_ticker_frame(w.ticker, db_path=db_path)
            raw_return_by_ticker_date[w.ticker] = dict(zip(frame["date"], frame["daily_return"]))
        y_true.append(w.y_raw)
        prior_value = raw_return_by_ticker_date[w.ticker][w.feature_end_date]
        y_prior_const.append(np.full(len(w.y_raw), prior_value, dtype=np.float64))

    y_true_arr = np.stack(y_true)
    y_prior_arr = np.stack(y_prior_const)
    zero_pred = np.zeros_like(y_true_arr)

    return {
        "zero": compute_metrics_per_step(y_true_arr, zero_pred),
        "prior": compute_metrics_per_step(y_true_arr, y_prior_arr),
    }
