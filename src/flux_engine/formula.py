"""
Composite flux-score formula for Phase E. Implements progress/flux_formula.md exactly
-- every constant/decision here is cited back to a section of that doc. Do not
change a number here without updating the doc (and vice versa); they must stay
in lockstep, same discipline as labeling/schema.py <-> progress/event_taxonomy.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

from src.propagation.decay import time_decay, DEFAULT_HALF_LIFE_DAYS
from src.propagation.graph import ExposureGraph, NODES, COUNTRY

# --- §3: source-reliability discount ---------------------------------------
# rel(e). gdelt_derived is discounted because GDELT rows have unconfirmed
# sector/company relevance (no article text); rule_based/manual are trusted at
# face value (real keyword-matched text); llm_assisted is an untested
# placeholder (no such rows exist yet in this environment).
RELIABILITY: dict[str, float] = {
    "manual": 1.00,
    "rule_based": 1.00,
    "llm_assisted": 0.85,   # doc §3: pure placeholder, no data to calibrate against
    "gdelt_derived": 0.50,  # doc §3: corridor-relevance discount
    "ollama_assisted": 0.65,  # doc §3: local free-model correction -- more content-aware
                              # than gdelt_derived's blind QuadClass heuristic (0.50), but a
                              # small local model with no calibration data, so kept below
                              # llm_assisted's hosted-frontier-model placeholder (0.85).
                              # Judgment call, not fit against real market reactions.
}

# --- §6: drop threshold + derived lookback window --------------------------
EPSILON = 0.01


def lookback_window_days(half_life_days: float = DEFAULT_HALF_LIFE_DAYS, epsilon: float = EPSILON) -> int:
    """doc §6: window = half_life * log2(1/epsilon), ceil'd."""
    return math.ceil(half_life_days * math.log2(1.0 / epsilon))


_CORRIDOR_COUNTRIES = frozenset(name for name, kind in NODES.items() if kind == COUNTRY)

# unclassified is excluded per doc §2, never queried against the graph here.
_EXCLUDED_EVENT_TYPES = frozenset({"unclassified"})


@dataclass
class ScoredEvent:
    """One event's contribution to one stock's flux score -- kept for
    transparency/debugging, not just the aggregate number."""
    event_id: str
    label_source: str
    event_type: str
    contribution: float          # c(e, s, t)
    polarity: Optional[float]
    imp: float
    tau: float
    days_since: float


@dataclass
class FluxResult:
    ticker: str
    flux_score: float
    direction: Optional[float]      # doc §4, None if no polarity-bearing event contributed
    direction_coverage: float       # doc §4, 0.0 if direction is None
    contributing_events: list[ScoredEvent]


def event_contribution(
    event_type: str,
    countries: Iterable[str],
    severity_score: float,
    confidence: float,
    label_source: str,
    published_at: datetime,
    stock: str,
    t: datetime,
    graph: ExposureGraph,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> tuple[float, float, float]:
    """
    doc §1: c(e,s,t) = sev * conf * rel * imp * tau.
    Returns (contribution, imp, tau) so callers can inspect intermediates
    without recomputing them (used by both flux_score() and the demo/tests).
    """
    if event_type in _EXCLUDED_EVENT_TYPES:  # doc §2
        return 0.0, 0.0, 0.0

    delta_days = (t - published_at).total_seconds() / 86400.0  # doc §6: calendar days (v1 choice)
    if delta_days < 0:  # doc §1 leakage guard: event must not be from the future
        return 0.0, 0.0, 0.0

    tau = time_decay(delta_days, half_life_days)
    if tau < EPSILON:  # doc §6
        return 0.0, 0.0, 0.0

    rel = RELIABILITY.get(label_source, 0.0)
    if rel == 0.0:
        return 0.0, 0.0, tau

    corridor = [c for c in countries if c in _CORRIDOR_COUNTRIES]  # doc §1 guard
    if not corridor:
        return 0.0, 0.0, tau

    exposures = graph.query(set(corridor), event_type)
    exp = exposures.get(stock)
    if exp is None:
        return 0.0, 0.0, tau
    imp = exp.impact

    contribution = severity_score * confidence * rel * imp * tau
    return contribution, imp, tau


def _cluster_key(event_type: str, countries: Iterable[str], published_at: datetime) -> tuple:
    """
    doc §5a: unit-of-evidence key for the day-level clustering fix. `imp` and
    `tau` are already fully determined by (corridor-countries, event_type) and
    calendar date respectively (doc §6 uses calendar days), so events sharing
    all three are, for this formula's purposes, the same piece of evidence --
    e.g. dozens of articles about one day's US-Iran escalation -- and must be
    collapsed to one representative before noisy-OR, not fed in individually.
    """
    corridor = tuple(sorted(c for c in countries if c in _CORRIDOR_COUNTRIES))
    return (published_at.date(), event_type, corridor)


def flux_score(
    events: list[dict],
    stock: str,
    t: datetime,
    graph: Optional[ExposureGraph] = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> FluxResult:
    """
    doc §5 + §5a: cluster same-day/same-corridor/same-event_type events to a
    single max-contribution representative first (same "max for correlated,
    noisy-OR for independent" principle Phase D already applies across graph
    paths within one event -- progress/exposure_graph.md §4 -- extended here across
    near-duplicate same-day articles), THEN noisy-OR aggregate across those
    representatives: 1 - prod(1 - c_i). Without this step, thousands of
    articles about one ongoing situation saturate every score to ~1.0 (see
    progress/flux_score_timeseries_findings.md) -- noisy-OR assumes independence,
    which same-day/same-corridor duplicates emphatically are not.

    `events` are dicts with keys: event_id, label_source, event_type,
    severity_score, confidence, polarity (optional), countries (list),
    published_at (datetime, tz-aware UTC).
    """
    graph = graph or ExposureGraph()

    # Step 1: per-event contribution, keyed by cluster.
    best_by_cluster: dict[tuple, tuple] = {}
    for e in events:
        c, imp, tau = event_contribution(
            event_type=e["event_type"],
            countries=e.get("countries", []),
            severity_score=e["severity_score"],
            confidence=e["confidence"],
            label_source=e["label_source"],
            published_at=e["published_at"],
            stock=stock,
            t=t,
            graph=graph,
            half_life_days=half_life_days,
        )
        if c <= 0.0:
            continue
        key = _cluster_key(e["event_type"], e.get("countries", []), e["published_at"])
        cur = best_by_cluster.get(key)
        if cur is None or c > cur[0]:
            best_by_cluster[key] = (c, imp, tau, e)

    # Step 2: noisy-OR + direction over the clustered representatives only --
    # same set for both, so a large duplicate cluster can't dominate direction
    # after being collapsed to one representative for magnitude.
    product = 1.0
    scored: list[ScoredEvent] = []
    pol_num = 0.0
    pol_den = 0.0
    total_c = 0.0

    for c, imp, tau, e in best_by_cluster.values():
        days_since = (t - e["published_at"]).total_seconds() / 86400.0
        scored.append(ScoredEvent(
            event_id=e["event_id"], label_source=e["label_source"],
            event_type=e["event_type"], contribution=c,
            polarity=e.get("polarity"), imp=imp, tau=tau, days_since=days_since,
        ))
        product *= (1.0 - c)
        total_c += c
        pol = e.get("polarity")
        if pol is not None:  # doc §4: only polarity-bearing events count toward direction
            pol_num += c * pol
            pol_den += c

    score = 1.0 - product
    direction = (pol_num / pol_den) if pol_den > 0 else None
    coverage = (pol_den / total_c) if total_c > 0 else 0.0

    return FluxResult(
        ticker=stock, flux_score=score, direction=direction,
        direction_coverage=coverage, contributing_events=scored,
    )


def flux_scores_for_stocks(
    events: list[dict],
    stocks: Iterable[str],
    t: datetime,
    graph: Optional[ExposureGraph] = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> dict[str, FluxResult]:
    graph = graph or ExposureGraph()
    return {s: flux_score(events, s, t, graph=graph, half_life_days=half_life_days) for s in stocks}
