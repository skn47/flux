"""
Event-centric endpoints: a global recent-events list, and a detail lookup
given a cluster-representative event_id (as surfaced by both the list below
and /api/tickers/{ticker}/events). The detail endpoint returns the event's
own descriptive metadata (from the precomputed event_catalog table) plus a
complete cross-ticker/cross-sector impact breakdown, computed live via
propagation.graph.ExposureGraph.query() -- a bounded, zero-I/O, in-memory
graph traversal, NOT flux_engine.query.load_events()'s 14-85s corpus scan.
This module never imports flux_engine or touches
classified_events/raw_events, matching the discipline in api/routers/flux.py
and api/README.md.

Only event_ids that were ever a cluster's max-K representative (i.e. appear
in event_catalog AND have at least one flux_attribution_daily row) are
queryable -- event_catalog rows are already cluster-representatives, so no
additional "is this notable" filtering is needed for the list endpoint, only
sorting/date-range filtering. The EXISTS join against flux_attribution_daily
guarantees every listed event_id also resolves on the detail endpoint below.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query

from api.config import EVENTS_LIST_DEFAULT_LIMIT
from api.db import query, query_one
from api.schemas import EventDetail, EventSummary, EventTickerImpact
from config.mvp_scope import TICKER_TO_SECTOR
from src.propagation.graph import ExposureGraph

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("", response_model=list[EventSummary])
def list_events(
    start: str | None = Query(default=None, description="YYYY-MM-DD, inclusive (matches published_at's date)"),
    end: str | None = Query(default=None, description="YYYY-MM-DD, inclusive (matches published_at's date)"),
    limit: int = Query(default=EVENTS_LIST_DEFAULT_LIMIT, ge=1, le=500),
) -> list[EventSummary]:
    sql = """
        SELECT c.event_id, c.event_type, c.label_source, c.severity_score,
               c.published_at, c.title,
               (SELECT corridor FROM flux_attribution_daily a
                WHERE a.event_id = c.event_id LIMIT 1) AS corridor
        FROM event_catalog c
        WHERE EXISTS (SELECT 1 FROM flux_attribution_daily a WHERE a.event_id = c.event_id)
    """
    params: list = []
    if start:
        sql += " AND c.published_at >= ?"
        params.append(start)
    if end:
        # published_at is a full ISO datetime; pad to end-of-day so same-day
        # events on `end` aren't excluded by a bare date-string comparison.
        sql += " AND c.published_at <= ?"
        params.append(f"{end}T23:59:59+00:00")
    sql += " ORDER BY c.published_at DESC LIMIT ?"
    params.append(limit)

    rows = query(sql, tuple(params))
    return [
        EventSummary(
            event_id=r["event_id"], event_type=r["event_type"], label_source=r["label_source"],
            severity_score=r["severity_score"], published_at=r["published_at"], title=r["title"],
            corridor=json.loads(r["corridor"]) if r["corridor"] else [],
        )
        for r in rows
    ]


@router.get("/{event_id}", response_model=EventDetail)
def event_detail(event_id: str) -> EventDetail:
    cat = query_one("SELECT * FROM event_catalog WHERE event_id = ?", (event_id,))
    if cat is None:
        raise HTTPException(status_code=404, detail=f"{event_id!r} is not a known event")

    attr = query_one(
        "SELECT DISTINCT corridor, event_type, k_value FROM flux_attribution_daily WHERE event_id = ? LIMIT 1",
        (event_id,),
    )
    if attr is None:
        raise HTTPException(status_code=404, detail=f"{event_id!r} has no attribution data")

    corridor = json.loads(attr["corridor"])
    exposures = ExposureGraph().query(set(corridor), attr["event_type"]) if corridor else {}

    top5_tickers = {
        r["ticker"]
        for r in query(
            "SELECT DISTINCT ticker FROM flux_attribution_daily WHERE event_id = ?", (event_id,)
        )
    }

    affected = sorted(
        (
            EventTickerImpact(
                ticker=t,
                sector=TICKER_TO_SECTOR[t],
                imp=exp.impact,
                path=exp.path,
                contribution=attr["k_value"] * exp.impact,
                ever_top5=t in top5_tickers,
            )
            for t, exp in exposures.items()
        ),
        key=lambda x: x.contribution,
        reverse=True,
    )

    return EventDetail(
        event_id=cat["event_id"],
        event_type=cat["event_type"],
        label_source=cat["label_source"],
        severity_score=cat["severity_score"],
        confidence=cat["confidence"],
        polarity=cat["polarity"],
        countries=json.loads(cat["countries"]),
        corridor=corridor,
        published_at=cat["published_at"],
        url=cat["url"],
        title=cat["title"],
        k_value=attr["k_value"],
        affected=affected,
    )
