"""Pydantic response models for the Phase H API. Field names/shapes mirror
the underlying tables directly (flux_scores_daily, flux_attribution_daily,
latest_predictions) -- these are thin serialization wrappers, not a separate
data model."""

from __future__ import annotations

from pydantic import BaseModel


class TickerOut(BaseModel):
    ticker: str
    sector: str
    company_names: list[str]


class SectorOut(BaseModel):
    sector: str
    tickers: list[str]


class FluxScorePoint(BaseModel):
    date: str
    flux_score: float
    direction: float | None
    direction_coverage: float
    n_clusters: int


class AttributionEvent(BaseModel):
    rank: int
    event_id: str
    label_source: str
    event_type: str
    corridor: list[str]
    path: list[str]
    cluster_date: str
    contribution: float
    imp: float
    tau: float
    days_since: float
    k_value: float


class PricePoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int


class EventDateMarker(BaseModel):
    date: str
    event_id: str
    event_type: str
    label_source: str
    corridor: list[str]


class EventTickerImpact(BaseModel):
    ticker: str
    sector: str
    imp: float
    path: list[str]
    contribution: float
    ever_top5: bool


class EventDetail(BaseModel):
    event_id: str
    event_type: str
    label_source: str
    severity_score: float
    confidence: float
    polarity: float | None
    countries: list[str]
    corridor: list[str]
    published_at: str
    url: str | None
    title: str | None
    k_value: float
    affected: list[EventTickerImpact]


class CausalGraphNode(BaseModel):
    id: str            # matches propagation/graph.py NODES keys / AttributionEvent.path entries
    node_type: str      # "country" | "sector" | "stock"
    is_focus: bool       # true only for the ticker this graph was requested for


class CausalGraphEdge(BaseModel):
    src: str
    dst: str
    weight: float
    channel: str | None
    note: str           # propagation.graph.Edge.note, "§anchor: " prefix stripped for display


class CausalGraphEvent(BaseModel):
    event_id: str
    rank: int
    event_type: str
    label_source: str
    title: str | None
    narrative: str | None     # null until narratives.run_precompute_narratives has reached it
    path: list[str]
    contribution: float
    cluster_date: str


class CausalGraphOut(BaseModel):
    ticker: str
    date: str
    nodes: list[CausalGraphNode]
    edges: list[CausalGraphEdge]
    events: list[CausalGraphEvent]


class EventSummary(BaseModel):
    event_id: str
    event_type: str
    label_source: str
    severity_score: float
    published_at: str
    title: str | None
    corridor: list[str]


class LatestPriceOut(BaseModel):
    ticker: str
    date: str
    close: float
    adj_close: float
    prior_close: float | None


class SparklinePoint(BaseModel):
    date: str
    adj_close: float


class SparklineOut(BaseModel):
    ticker: str
    points: list[SparklinePoint]


class SimulateTickerImpact(BaseModel):
    ticker: str
    sector: str
    imp: float
    path: list[str]
    contribution: float


class SimulateResult(BaseModel):
    event_type: str
    countries: list[str]
    severity: float
    confidence: float
    affected: list[SimulateTickerImpact]


class PredictionOut(BaseModel):
    ticker: str
    variant: str
    label_date: str
    prior_date: str
    actual_return_L: float | None
    predicted_return_L: float
    computed_at: str


class ForecastPointOut(BaseModel):
    ticker: str
    variant: str
    as_of_date: str
    horizon_step: int
    target_date: str
    point_price: float
    lo_price: float
    hi_price: float
    computed_at: str


class HealthOut(BaseModel):
    status: str
    db_path_ok: bool
    latest_flux_score_computed_at: str | None
