"""
Latest stored test-split prediction per ticker/variant, from the
`latest_predictions` table (populated by backtest/persist_predictions.py).

Deliberately NOT live inference -- see the Phase H plan's locked decision #3:
`features_daily`'s test split already tracks through the most recent
ingested trading day, so "latest test-split prediction" is a reasonable
stand-in for "current," refreshed only when the pipeline is manually
rerun. This endpoint never imports torch/lstm -- a real decoupling: the API
process only ever does an indexed SELECT against a tiny table.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.db import query
from api.schemas import PredictionOut
from config.mvp_scope import TRACKED_TICKERS

router = APIRouter(prefix="/api/tickers", tags=["predictions"])


def _require_tracked(ticker: str) -> None:
    if ticker not in TRACKED_TICKERS:
        raise HTTPException(status_code=404, detail=f"{ticker!r} is not a tracked ticker")


@router.get("/{ticker}/prediction", response_model=list[PredictionOut])
def latest_prediction(ticker: str) -> list[PredictionOut]:
    _require_tracked(ticker)

    out: list[PredictionOut] = []
    for variant in ("baseline", "flux"):
        rows = query(
            """
            SELECT ticker, variant, label_date, prior_date, actual_return_L,
                   predicted_return_L, computed_at
            FROM latest_predictions
            WHERE ticker = ? AND variant = ?
            ORDER BY label_date DESC
            LIMIT 1
            """,
            (ticker, variant),
        )
        if rows:
            r = rows[0]
            out.append(PredictionOut(
                ticker=r["ticker"], variant=r["variant"], label_date=r["label_date"],
                prior_date=r["prior_date"], actual_return_L=r["actual_return_L"],
                predicted_return_L=r["predicted_return_L"], computed_at=r["computed_at"],
            ))
    return out


@router.get("/{ticker}/predictions/history", response_model=list[PredictionOut])
def prediction_history(
    ticker: str,
    variant: str | None = Query(default=None, description="baseline or flux; omit for both"),
) -> list[PredictionOut]:
    """Full historical range of stored test-split predictions -- unlike
    latest_prediction above, this does not LIMIT 1. Powers the AI Horizon's
    predicted-point display, the time-travel slider's date domain, and
    client-side directional-accuracy computation (sign(predicted) ==
    sign(actual)) -- no new metric needs storing for that, it's derived from
    this response."""
    _require_tracked(ticker)
    variants = [variant] if variant else ["baseline", "flux"]
    out: list[PredictionOut] = []
    for v in variants:
        rows = query(
            """
            SELECT ticker, variant, label_date, prior_date, actual_return_L,
                   predicted_return_L, computed_at
            FROM latest_predictions
            WHERE ticker = ? AND variant = ?
            ORDER BY label_date ASC
            """,
            (ticker, v),
        )
        out.extend(
            PredictionOut(
                ticker=r["ticker"], variant=r["variant"], label_date=r["label_date"],
                prior_date=r["prior_date"], actual_return_L=r["actual_return_L"],
                predicted_return_L=r["predicted_return_L"], computed_at=r["computed_at"],
            )
            for r in rows
        )
    return out
