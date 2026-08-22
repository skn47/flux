"""
Flux-score history, price history, and event drill-down for one ticker.

Every endpoint reads ONLY from small, indexed, precomputed tables
(flux_scores_daily, flux_attribution_daily, daily_prices) -- never
flux_engine.formula.flux_score()/load_events(), which require a fresh
classified_events JOIN raw_events scan measured at 14-85s per call against
the live 6.4M-row corpus. See api/README.md and the Phase H plan section
("Investigation before designing") for why that path is not servable
synchronously.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

from fastapi import APIRouter, HTTPException, Query

from api.db import query, query_one
from api.schemas import (
    AttributionEvent,
    CausalGraphEdge,
    CausalGraphEvent,
    CausalGraphNode,
    CausalGraphOut,
    EventDateMarker,
    LatestPriceOut,
    PricePoint,
    FluxScorePoint,
    SparklineOut,
    SparklinePoint,
)
from config.mvp_scope import TRACKED_TICKERS
from src.propagation.graph import EDGES_BY_PAIR, ExposureGraph

router = APIRouter(prefix="/api/tickers", tags=["flux"])


def _require_tracked(ticker: str) -> None:
    if ticker not in TRACKED_TICKERS:
        raise HTTPException(status_code=404, detail=f"{ticker!r} is not a tracked ticker")


@router.get("/{ticker}/flux-scores", response_model=list[FluxScorePoint])
def flux_scores(
    ticker: str,
    start: str | None = Query(default=None, description="YYYY-MM-DD, inclusive"),
    end: str | None = Query(default=None, description="YYYY-MM-DD, inclusive"),
) -> list[FluxScorePoint]:
    _require_tracked(ticker)
    sql = "SELECT date, flux_score, direction, direction_coverage, n_clusters FROM flux_scores_daily WHERE ticker = ?"
    params: list = [ticker]
    if start:
        sql += " AND date >= ?"
        params.append(start)
    if end:
        sql += " AND date <= ?"
        params.append(end)
    sql += " ORDER BY date ASC"
    rows = query(sql, tuple(params))
    return [
        FluxScorePoint(
            date=r["date"], flux_score=r["flux_score"], direction=r["direction"],
            direction_coverage=r["direction_coverage"], n_clusters=r["n_clusters"],
        )
        for r in rows
    ]


# yfinance's period="max" pull goes back to each ticker's IPO -- unbounded,
# that's 20-30+ years of daily bars (1.2MB+ of JSON for a ticker like NVDA or
# INTC). A candlestick chart has no use for pre-2000 data by default, and an
# unbounded-by-default public endpoint is a footgun regardless of caller
# behavior. Callers that actually want full history can still pass an
# explicit `start`.
DEFAULT_LOOKBACK_DAYS = 730


@router.get("/{ticker}/prices", response_model=list[PricePoint])
def prices(
    ticker: str,
    start: str | None = Query(default=None, description="YYYY-MM-DD, inclusive"),
    end: str | None = Query(default=None, description="YYYY-MM-DD, inclusive"),
) -> list[PricePoint]:
    _require_tracked(ticker)
    if start is None and end is None:
        row = query_one("SELECT MAX(date) AS latest FROM daily_prices WHERE ticker = ?", (ticker,))
        latest = row["latest"] if row else None
        if latest:
            start = (date.fromisoformat(latest) - timedelta(days=DEFAULT_LOOKBACK_DAYS)).isoformat()
    sql = (
        "SELECT date, open, high, low, close, adj_close, volume "
        "FROM daily_prices WHERE ticker = ?"
    )
    params: list = [ticker]
    if start:
        sql += " AND date >= ?"
        params.append(start)
    if end:
        sql += " AND date <= ?"
        params.append(end)
    sql += " ORDER BY date ASC"
    rows = query(sql, tuple(params))
    return [
        PricePoint(
            date=r["date"], open=r["open"], high=r["high"], low=r["low"],
            close=r["close"], adj_close=r["adj_close"], volume=r["volume"],
        )
        for r in rows
    ]


@router.get("/latest-prices", response_model=list[LatestPriceOut])
def latest_prices() -> list[LatestPriceOut]:
    """Bulk latest known close per tracked ticker, plus the prior close so a
    caller can show a day-over-day delta -- powers the ticker tape. A
    single-segment literal path, so it never collides with the
    /{ticker}/... routes above (those always require a second path segment).

    Deliberately a per-ticker LIMIT 2 loop rather than one clever
    LAG/ROW_NUMBER window-function query over the full 225K-row table:
    measured, the window-function form took ~400ms (SQLite plans each of
    LAG and ROW_NUMBER as its own pass needing a temp B-tree sort, 3 sorts
    total, even with the (ticker, date) primary key already in the right
    order) versus low-single-digit ms per ticker here, an indexed range
    scan on the same primary key. Same "simple, obviously-correct SQL over
    23 small queries" preference already used by sparklines() below.
    """
    out: list[LatestPriceOut] = []
    for ticker in sorted(TRACKED_TICKERS):
        rows = query(
            "SELECT date, close, adj_close FROM daily_prices WHERE ticker = ? "
            "ORDER BY date DESC LIMIT 2",
            (ticker,),
        )
        if not rows:
            continue
        latest = rows[0]
        prior_close = rows[1]["close"] if len(rows) > 1 else None
        out.append(
            LatestPriceOut(
                ticker=ticker, date=latest["date"], close=latest["close"],
                adj_close=latest["adj_close"], prior_close=prior_close,
            )
        )
    return out


@router.get("/sparklines", response_model=list[SparklineOut])
def sparklines(days: int = Query(default=30, ge=5, le=180)) -> list[SparklineOut]:
    """Last `days` adj_close points per tracked ticker -- powers the sector
    matrix's mini sparklines. One small indexed per-ticker query each (a PK
    seek on (ticker,date)) rather than one clever multi-ticker query, matching
    this codebase's existing preference for simple, obviously-correct SQL."""
    out: list[SparklineOut] = []
    for ticker in TRACKED_TICKERS:
        rows = query(
            "SELECT date, adj_close FROM daily_prices WHERE ticker = ? ORDER BY date DESC LIMIT ?",
            (ticker, days),
        )
        points = [SparklinePoint(date=r["date"], adj_close=r["adj_close"]) for r in reversed(rows)]
        out.append(SparklineOut(ticker=ticker, points=points))
    return out


@router.get("/{ticker}/event-dates", response_model=list[EventDateMarker])
def event_dates(
    ticker: str,
    start: str | None = Query(default=None, description="YYYY-MM-DD, inclusive"),
    end: str | None = Query(default=None, description="YYYY-MM-DD, inclusive"),
) -> list[EventDateMarker]:
    """Dates where the top-ranked (rank=1) contributor is ALSO a same-day-new
    event (days_since=0) -- i.e. the day a new event entered this ticker's
    attribution, not a day it's merely still decaying in the trailing window.
    Drives the price chart's click-through markers."""
    _require_tracked(ticker)
    sql = (
        "SELECT date, event_id, event_type, label_source, corridor "
        "FROM flux_attribution_daily WHERE ticker = ? AND rank = 1 AND days_since = 0"
    )
    params: list = [ticker]
    if start:
        sql += " AND date >= ?"
        params.append(start)
    if end:
        sql += " AND date <= ?"
        params.append(end)
    sql += " ORDER BY date ASC"
    rows = query(sql, tuple(params))
    return [
        EventDateMarker(
            date=r["date"], event_id=r["event_id"], event_type=r["event_type"],
            label_source=r["label_source"], corridor=json.loads(r["corridor"]),
        )
        for r in rows
    ]


@router.get("/{ticker}/events", response_model=list[AttributionEvent])
def events(
    ticker: str,
    date: str | None = Query(default=None, description="YYYY-MM-DD; defaults to the latest scored date"),
) -> list[AttributionEvent]:
    _require_tracked(ticker)
    if date is None:
        row = query_one(
            "SELECT MAX(date) AS latest FROM flux_scores_daily WHERE ticker = ?", (ticker,)
        )
        date = row["latest"] if row else None
        if date is None:
            return []

    rows = query(
        """
        SELECT rank, event_id, label_source, event_type, corridor, path,
               cluster_date, contribution, imp, tau, days_since, k_value
        FROM flux_attribution_daily
        WHERE ticker = ? AND date = ?
        ORDER BY rank ASC
        """,
        (ticker, date),
    )
    return [
        AttributionEvent(
            rank=r["rank"], event_id=r["event_id"], label_source=r["label_source"],
            event_type=r["event_type"], corridor=json.loads(r["corridor"]),
            path=json.loads(r["path"]), cluster_date=r["cluster_date"],
            contribution=r["contribution"], imp=r["imp"], tau=r["tau"],
            days_since=r["days_since"], k_value=r["k_value"],
        )
        for r in rows
    ]


def _clean_note(note: str) -> str:
    """Strip the leading '§anchor: ' prefix propagation/graph.py's Edge.note
    carries -- the anchor is only meaningful as a progress/exposure_graph.md
    cross-ref, not for display."""
    if note.startswith("§") and ": " in note:
        return note.split(": ", 1)[1]
    return note


def _narratives_table_exists() -> bool:
    row = query_one(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'event_narratives'"
    )
    return row is not None


@router.get("/{ticker}/causal-graph", response_model=CausalGraphOut)
def causal_graph(
    ticker: str,
    date: str | None = Query(default=None, description="YYYY-MM-DD; defaults to the latest scored date"),
) -> CausalGraphOut:
    """
    Deduped node/edge graph + per-event narrative for the 3D causality scene
    (frontend/src/components/CausalityScene3D.tsx), covering the same top-5
    daily attribution rows `events()` above returns for this ticker/date, but
    reshaped as a graph rather than a flat per-event list, and enriched with:
    real edge metadata (weight/channel/note) from propagation.graph.EDGES_BY_PAIR
    (zero-I/O, in-memory -- same discipline as api/routers/events.py's live
    ExposureGraph use), and an AI-generated narrative per event from the
    separately-populated event_narratives table (narratives/run_precompute_narratives.py)
    -- null until that batch job reaches a given event, a first-class, always-
    valid response state, not an error.
    """
    _require_tracked(ticker)
    if date is None:
        row = query_one("SELECT MAX(date) AS latest FROM flux_scores_daily WHERE ticker = ?", (ticker,))
        date = row["latest"] if row else None
        if date is None:
            return CausalGraphOut(ticker=ticker, date="", nodes=[], edges=[], events=[])

    has_narratives = _narratives_table_exists()
    sql = """
        SELECT a.rank, a.event_id, a.label_source, a.event_type, a.path,
               a.cluster_date, a.contribution, c.title
               {narrative_col}
        FROM flux_attribution_daily a
        JOIN event_catalog c ON c.event_id = a.event_id
        {narrative_join}
        WHERE a.ticker = ? AND a.date = ?
        ORDER BY a.rank ASC
    """.format(
        narrative_col=", n.narrative" if has_narratives else ", NULL AS narrative",
        narrative_join="LEFT JOIN event_narratives n ON n.event_id = a.event_id" if has_narratives else "",
    )
    rows = query(sql, (ticker, date))

    graph = ExposureGraph()
    events_out: list[CausalGraphEvent] = []
    node_ids: dict[str, str] = {}  # id -> node_type
    edges_out: list[CausalGraphEdge] = []
    seen_edges: set[tuple[str, str, str | None]] = set()

    for r in rows:
        path: list[str] = json.loads(r["path"])
        events_out.append(
            CausalGraphEvent(
                event_id=r["event_id"], rank=r["rank"], event_type=r["event_type"],
                label_source=r["label_source"], title=r["title"], narrative=r["narrative"],
                path=path, contribution=r["contribution"], cluster_date=r["cluster_date"],
            )
        )
        for name in path:
            node_ids.setdefault(name, graph.node_type(name))
        for src, dst in zip(path, path[1:]):
            for edge in EDGES_BY_PAIR.get((src, dst), []):
                key = (src, dst, edge.channel)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edges_out.append(
                    CausalGraphEdge(
                        src=src, dst=dst, weight=edge.weight, channel=edge.channel,
                        note=_clean_note(edge.note),
                    )
                )

    nodes_out = [
        CausalGraphNode(id=name, node_type=node_type, is_focus=(name == ticker))
        for name, node_type in node_ids.items()
    ]

    return CausalGraphOut(ticker=ticker, date=date, nodes=nodes_out, edges=edges_out, events=events_out)
