"""Sector/ticker listing (in-memory, from config.mvp_scope -- no DB touch)
and a health check."""

from __future__ import annotations

from fastapi import APIRouter

from api.db import query_one
from api.schemas import HealthOut, SectorOut, TickerOut
from config.mvp_scope import SECTORS, TRACKED_STOCKS, TICKER_TO_SECTOR

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    db_ok = True
    latest = None
    try:
        row = query_one("SELECT MAX(computed_at) AS latest FROM flux_scores_daily")
        latest = row["latest"] if row else None
    except Exception:
        db_ok = False
    return HealthOut(status="ok" if db_ok else "degraded", db_path_ok=db_ok,
                      latest_flux_score_computed_at=latest)


@router.get("/sectors", response_model=list[SectorOut])
def list_sectors() -> list[SectorOut]:
    return [
        SectorOut(sector=name, tickers=[s["ticker"] for s in sector["stocks"]])
        for name, sector in SECTORS.items()
    ]


@router.get("/tickers", response_model=list[TickerOut])
def list_tickers() -> list[TickerOut]:
    return [
        TickerOut(
            ticker=stock["ticker"],
            sector=TICKER_TO_SECTOR[stock["ticker"]],
            company_names=stock["company_names"],
        )
        for stock in TRACKED_STOCKS
    ]
