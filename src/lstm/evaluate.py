"""
Prediction-accuracy metrics only -- MAE, RMSE, MAPE -- computed on raw
(unscaled) units. No backtesting, no trading-signal conversion, no
Sharpe/drawdown here; that is a separate later phase (Phase G) and
explicitly out of scope for this package (see task spec point 7).

R1 REWORK (see lstm/dataset.py's module docstring): the target is now
`daily_return`, not `adj_close`, so these metrics are computed on raw RETURN
units (e.g. 0.02 = 2%/day), not raw price units (e.g. $450). This changes
MAPE's behavior in an important, honestly-flagged way: MAPE divides by
`y_true`, and a near-zero actual daily return (common -- returns cluster
near 0) makes that day's percentage error blow up or become numerically
enormous, even for a model with a genuinely small absolute error. This is a
REAL statistical property of MAPE-on-returns, not a bug -- unlike
MAPE-on-price-level (the old target), which was well-behaved because
`adj_close` is never near zero. MAE and RMSE (scale-dependent but not
divide-by-near-zero-prone) are therefore treated as the PRIMARY headline
metrics for the R1 return target; MAPE is still reported (for continuity)
but should be read with this caveat -- see lstm/README.md's "MAPE-on-returns
caveat" section.

MAPE uses the standard x100 convention (spec_sheet.md flags Paper 1's
formula as missing the x100 in its typeset equation despite reporting
percentages -- this module applies it explicitly, matching what both papers
actually report).

Two independent implementations are provided on purpose:
  - `compute_metrics` -- vectorized numpy, used by the training/eval pipeline.
  - `recompute_metrics_independently` -- plain-Python loops, no numpy
    vectorization, used ONLY as a second, independently-coded check before
    numbers are written into lstm/README.md (this project's established
    verification discipline -- see features/README.md's "Verification"
    section for the same practice applied to the previous pipeline step).
Both must agree to float precision or something is wrong.

`persistence_baseline_metrics` (new, R1): two trivial, model-free baselines
computed directly from raw test-split return data -- "predict return = 0"
and "predict return = yesterday's realized return" -- so a reader can see
how much (if anything) the LSTM earns over doing nothing / naive momentum
continuation. Cheap to add now that the target is already a return (was
flagged as a missing gap in lstm/README.md when the target was still a price
level).
"""

from __future__ import annotations

import math

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    R1 note (see lstm/evaluate.py's module docstring): with a `daily_return`
    target, `y_true` can legitimately be exactly 0.0 on a real flat-return
    day (unlike the old `adj_close` target, which was never zero). MAPE
    divides by `y_true`, so this produces `inf` (0 numerator would give
    `nan`) rather than a crash -- `np.errstate` below suppresses numpy's
    printed RuntimeWarning for this EXPECTED, already-documented condition
    (not silencing an unexpected one); the resulting inf/nan is left in the
    output, not hidden, and `assert_metrics_agree` below explicitly handles
    the nan case when comparing against the independent recompute.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = math.sqrt(np.mean(err ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs(err / y_true)) * 100.0
    return {"n": int(len(y_true)), "mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def recompute_metrics_independently(y_true: list[float], y_pred: list[float]) -> dict:
    """Deliberately non-vectorized re-implementation, used as an independent
    cross-check of compute_metrics() before reporting numbers.

    R1 note: `t == 0` is handled explicitly (not left to crash on a
    ZeroDivisionError) to match numpy's IEEE-754 float division semantics
    used by compute_metrics() above -- `x/0 -> inf` (x != 0), `0/0 -> nan` --
    so the two independent implementations agree even on this edge case,
    which a `daily_return` target can hit for real (see compute_metrics'
    docstring) and the old `adj_close` target never could."""
    n = len(y_true)
    if n == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}
    abs_err_sum = 0.0
    sq_err_sum = 0.0
    pct_err_sum = 0.0
    for t, p in zip(y_true, y_pred):
        e = t - p
        abs_err_sum += abs(e)
        sq_err_sum += e * e
        if t == 0.0:
            pct_err_sum += float("nan") if e == 0.0 else float("inf")
        else:
            pct_err_sum += abs(e / t)
    mae = abs_err_sum / n
    rmse = math.sqrt(sq_err_sum / n)
    mape = (pct_err_sum / n) * 100.0
    return {"n": n, "mae": mae, "rmse": rmse, "mape": mape}


def persistence_baseline_metrics(y_true: np.ndarray, y_prior: np.ndarray) -> dict:
    """
    Two trivial, model-free baselines on raw daily_return units, both
    verified via the same vectorized-vs-independent-recompute pattern as the
    LSTM's own metrics above:
      - "zero"  -- predict next-day return = 0 every day.
      - "prior" -- predict next-day return = the realized return on the
        immediately preceding trading day (a naive persistence/momentum-
        continuation baseline; NOT "predict tomorrow's price = today's
        price", which was the old price-level baseline's natural analogue --
        here the analogous naive rule for a RETURN target is "tomorrow's
        return = today's return", since "tomorrow's return = 0" is the
        return-target equivalent of a flat prediction).
    Caller supplies `y_prior` (raw daily_return at each window's
    feature_end_date, i.e. L-1) -- this function does no DB access itself.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prior = np.asarray(y_prior, dtype=np.float64)
    zero_pred = np.zeros_like(y_true)

    m_zero_vec = compute_metrics(y_true, zero_pred)
    m_zero_indep = recompute_metrics_independently(y_true.tolist(), zero_pred.tolist())
    assert_metrics_agree(m_zero_vec, m_zero_indep)

    m_prior_vec = compute_metrics(y_true, y_prior)
    m_prior_indep = recompute_metrics_independently(y_true.tolist(), y_prior.tolist())
    assert_metrics_agree(m_prior_vec, m_prior_indep)

    return {"zero": m_zero_vec, "prior": m_prior_vec}


def assert_metrics_agree(a: dict, b: dict, tol: float = 1e-6) -> None:
    """R1 note: MAPE on a `daily_return` target can legitimately be `nan`
    (0/0, an exact-zero prediction on an exact-zero-return day) or `inf`
    (x/0, any nonzero error on an exact-zero-return day) -- both handled
    explicitly below (not just nan), since `inf - inf = nan` would otherwise
    make the plain `abs(a-b) < tol` check spuriously fail even when both
    implementations agree on "infinite"."""
    assert a["n"] == b["n"], f"n mismatch: {a['n']} vs {b['n']}"
    for key in ("mae", "rmse", "mape"):
        if math.isnan(a[key]) and math.isnan(b[key]):
            continue
        if math.isinf(a[key]) and math.isinf(b[key]) and a[key] == b[key]:
            continue
        assert abs(a[key] - b[key]) < tol, (
            f"{key} mismatch between vectorized and independent recompute: "
            f"{a[key]} vs {b[key]}"
        )
