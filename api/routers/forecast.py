"""
Latest stored seq2seq 30-day-forward, conformal-calibrated price forecast per
ticker, from the `seq2seq_forecast` table (populated by
backtest/persist_seq2seq_forecast.py).

Same decoupling as api/routers/predictions.py: this endpoint never imports
torch/lstm/backtest -- the API process only ever does an indexed SELECT
against a small table. Only the flux-conditioned ("flux") variant is stored
today, so `variant` defaults to it, but the parameter is kept (rather than
hardcoded into the query) since the table's schema already supports more.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.db import query
from api.schemas import ForecastPointOut
from config.mvp_scope import TRACKED_TICKERS

router = APIRouter(prefix="/api/tickers", tags=["forecast"])


def _require_tracked(ticker: str) -> None:
    if ticker not in TRACKED_TICKERS:
        raise HTTPException(status_code=404, detail=f"{ticker!r} is not a tracked ticker")


@router.get("/{ticker}/forecast", response_model=list[ForecastPointOut])
def latest_forecast(
    ticker: str,
    variant: str = Query(default="flux", description="Forecast variant; only 'flux' is populated today"),
) -> list[ForecastPointOut]:
    _require_tracked(ticker)
    rows = query(
        """
        SELECT ticker, variant, as_of_date, horizon_step, target_date,
               point_price, lo_price, hi_price, computed_at
        FROM seq2seq_forecast
        WHERE ticker = ? AND variant = ? AND as_of_date = (
            SELECT MAX(as_of_date) FROM seq2seq_forecast WHERE ticker = ? AND variant = ?
        )
        ORDER BY horizon_step ASC
        """,
        (ticker, variant, ticker, variant),
    )
    return [
        ForecastPointOut(
            ticker=r["ticker"], variant=r["variant"], as_of_date=r["as_of_date"],
            horizon_step=r["horizon_step"], target_date=r["target_date"],
            point_price=r["point_price"], lo_price=r["lo_price"], hi_price=r["hi_price"],
            computed_at=r["computed_at"],
        )
        for r in rows
    ]
