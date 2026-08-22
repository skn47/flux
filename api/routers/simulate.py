"""
Stateless what-if event simulator. Recomputes per-ticker flux/impact via the
same pure, zero-I/O ExposureGraph traversal already used live by
GET /api/events/{event_id} (see propagation/graph.py's module docstring) --
NOT flux_engine.query.load_events()'s corpus scan, and NOT LSTM inference.
This module never imports flux_engine or lstm -- see api/README.md's "never
runs LSTM inference in a request handler" principle. GET-only (matches
api/main.py's CORS allow_methods=["GET"]).

The formula below mirrors flux_engine/formula.py's single-event contribution
shape (severity * confidence * reliability * imp * tau) inlined rather than
imported, keeping this route's only import from propagation.graph (matching
events.py's existing precedent). A hypothetical event is treated as
maximally reliable (reliability=1.0, i.e. "manual"/synthetic input) and
"right now" (tau=1.0, zero decay).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.schemas import SimulateResult, SimulateTickerImpact
from config.mvp_scope import TICKER_TO_SECTOR
from src.propagation.graph import COUNTRY, EVENT_CHANNEL_ACTIVATION, NODES, ExposureGraph

router = APIRouter(prefix="/api/simulate", tags=["simulate"])

VALID_COUNTRIES = sorted(n for n, k in NODES.items() if k == COUNTRY)
VALID_EVENT_TYPES = sorted(set(EVENT_CHANNEL_ACTIVATION) - {"unclassified"})
_RELIABILITY_MANUAL = 1.0


@router.get("", response_model=SimulateResult)
def simulate(
    event_type: str = Query(...),
    countries: list[str] = Query(..., description="repeat param, e.g. ?countries=Taiwan&countries=South Korea"),
    severity: float = Query(default=0.7, ge=0.0, le=1.0),
    confidence: float = Query(default=0.7, ge=0.0, le=1.0),
) -> SimulateResult:
    if event_type not in VALID_EVENT_TYPES:
        raise HTTPException(status_code=400, detail=f"event_type must be one of {VALID_EVENT_TYPES}")
    bad = [c for c in countries if c not in VALID_COUNTRIES]
    if bad:
        raise HTTPException(status_code=400, detail=f"unknown countries {bad}; valid: {VALID_COUNTRIES}")

    exposures = ExposureGraph().query(set(countries), event_type)
    affected = sorted(
        (
            SimulateTickerImpact(
                ticker=t, sector=TICKER_TO_SECTOR[t], imp=exp.impact, path=exp.path,
                contribution=severity * confidence * _RELIABILITY_MANUAL * exp.impact,
            )
            for t, exp in exposures.items()
        ),
        key=lambda x: x.contribution,
        reverse=True,
    )
    return SimulateResult(
        event_type=event_type, countries=countries, severity=severity,
        confidence=confidence, affected=affected,
    )
