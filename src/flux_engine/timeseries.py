"""
Daily flux_score(s, t) time series for Phase F.

Why not just call flux_engine.formula.flux_score once per output date
------------------------------------------------------------------------
That's the obviously-correct approach and is what progress/flux_formula.md
specifies -- but a single date's flux_engine.query.load_events() call takes
50-85s against the ~6.4M-row backfilled corpus (see
progress/flux_score_timeseries_findings.md). A ~2-year daily series computed
that way would take multiple hours. This module computes the exact same
formula (doc §5a: cluster by (date, event_type, corridor), keep the max-c
representative per cluster, noisy-OR across clusters), just organized for
bulk computation:

1. Bulk-load the full non-unclassified event set ONCE (not once per date).
2. Pre-cluster GLOBALLY, once, rather than re-clustering inside every
   per-date window. This is NOT an approximation of doc §5a -- within one
   cluster (fixed date, event_type, corridor), imp(e,s) and tau(e,t) are
   identical for every event in it (both are fully determined by
   corridor+event_type+date, i.e. the cluster key itself). So for ANY fixed
   (s, t), ranking events in a cluster by c(e,s,t) = K(e)*imp(e,s)*tau(e,t)
   reduces to ranking by K(e) = sev(e)*conf(e)*rel(e) alone, since
   imp(e,s)*tau(e,t) is the same positive multiplier for every e in the
   cluster. The max-c representative is therefore always the max-K
   representative, independent of s and t -- so it can be picked once,
   globally, instead of re-derived inside every date's window query.
3. Per output date, noisy-OR only over the clusters whose date falls in the
   trailing lookback window -- a small, bounded set (~100-700 clusters),
   not the raw ~60k-90k events in that window.

Verified equivalent to flux_engine.formula.flux_score on the same inputs
(see the module's own self-check in flux_engine/README.md).
"""

from __future__ import annotations

import bisect
import heapq
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from src.ingestion.storage import DEFAULT_DB_PATH
from src.propagation.decay import DEFAULT_HALF_LIFE_DAYS, time_decay
from src.propagation.graph import ExposureGraph
from src.flux_engine.formula import EPSILON, RELIABILITY, _CORRIDOR_COUNTRIES, lookback_window_days
from src.flux_engine.query import load_events


@dataclass
class Cluster:
    cluster_date: date
    event_type: str
    corridor: tuple
    K: float                      # sev * conf * rel of the representative event
    imp: dict                     # ticker -> impact, precomputed once via ExposureGraph.query
    polarity: Optional[float]     # representative event's polarity, if any
    representative_event_id: str
    label_source: str = ""        # representative event's label_source (Phase H attribution)
    paths: dict = None            # ticker -> path (list[str]), from StockExposure.path (Phase H attribution)


def build_clusters(events: list[dict], graph: Optional[ExposureGraph] = None) -> list[Cluster]:
    """doc §5a clustering, applied globally (see module docstring for why this
    is exact, not approximate). `events` must already exclude `unclassified`
    (flux_engine.query.load_events does this at the SQL level)."""
    graph = graph or ExposureGraph()
    best: dict[tuple, tuple] = {}

    for e in events:
        rel = RELIABILITY.get(e["label_source"], 0.0)
        if rel == 0.0:
            continue
        corridor = tuple(sorted(c for c in e.get("countries", []) if c in _CORRIDOR_COUNTRIES))
        if not corridor:
            continue
        K = e["severity_score"] * e["confidence"] * rel
        key = (e["published_at"].date(), e["event_type"], corridor)
        cur = best.get(key)
        if cur is None or K > cur[0]:
            best[key] = (K, e)

    query_cache: dict[tuple, dict] = {}
    clusters: list[Cluster] = []
    for (cluster_date, event_type, corridor), (K, e) in best.items():
        cache_key = (corridor, event_type)
        if cache_key not in query_cache:
            query_cache[cache_key] = graph.query(set(corridor), event_type)
        imp_map = query_cache[cache_key]
        imp = {ticker: exp.impact for ticker, exp in imp_map.items()}
        paths = {ticker: exp.path for ticker, exp in imp_map.items()}
        clusters.append(Cluster(
            cluster_date=cluster_date, event_type=event_type, corridor=corridor,
            K=K, imp=imp, polarity=e.get("polarity"),
            representative_event_id=e["event_id"],
            label_source=e["label_source"], paths=paths,
        ))

    clusters.sort(key=lambda c: c.cluster_date)
    return clusters


@dataclass
class AttributionRow:
    """One contributing cluster for one (ticker, day), Phase H drill-down data.
    Values are taken from the exact same contribution/imp/tau/days_since
    already computed for the noisy-OR product below -- not recomputed."""
    event_id: str
    label_source: str
    event_type: str
    corridor: tuple
    path: list
    cluster_date: date
    contribution: float
    imp: float
    tau: float
    days_since: int
    k_value: float


@dataclass
class DailyScore:
    day: date
    flux_score: float
    direction: Optional[float]
    direction_coverage: float
    n_clusters: int
    top_contributors: list = None  # list[AttributionRow], highest-contribution first, len <= top_k


def compute_daily_series(
    clusters: list[Cluster],
    stocks: list[str],
    start: date,
    end: date,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    epsilon: float = EPSILON,
    top_k: int = 5,
) -> dict[str, list[DailyScore]]:
    """One flux_score(s, t) per calendar day in [start, end] for each stock,
    via a sliding window over pre-sorted clusters (two-pointer, amortized
    O(total_days + total_clusters) rather than O(days * clusters)).

    Also tracks the top-`top_k` contributing clusters per (stock, day) via a
    bounded min-heap (O(log top_k) per candidate, not a full sort of the
    window) -- Phase H attribution data, populated from values already
    computed for the noisy-OR product itself, not a second pass."""
    window_days = lookback_window_days(half_life_days, epsilon)
    cluster_dates = [c.cluster_date for c in clusters]
    results: dict[str, list[DailyScore]] = {s: [] for s in stocks}

    lo = 0
    hi = 0
    n_days = (end - start).days + 1
    for i in range(n_days):
        t = start + timedelta(days=i)
        window_start = t - timedelta(days=window_days)
        lo = bisect.bisect_left(cluster_dates, window_start, lo)
        hi = bisect.bisect_right(cluster_dates, t, hi)
        window = clusters[lo:hi]

        for s in stocks:
            product = 1.0
            pol_num = 0.0
            pol_den = 0.0
            total_c = 0.0
            n_contrib = 0
            top_heap: list = []  # min-heap of (contribution, counter, cluster, imp, tau, days_since)
            counter = 0
            for c in window:
                imp = c.imp.get(s)
                if not imp:
                    continue
                days_since = (t - c.cluster_date).days
                tau = time_decay(days_since, half_life_days)
                if tau < epsilon:
                    continue
                contribution = c.K * imp * tau
                if contribution <= 0.0:
                    continue
                product *= (1.0 - contribution)
                total_c += contribution
                n_contrib += 1
                if c.polarity is not None:
                    pol_num += contribution * c.polarity
                    pol_den += contribution

                counter += 1
                entry = (contribution, counter, c, imp, tau, days_since)
                if len(top_heap) < top_k:
                    heapq.heappush(top_heap, entry)
                elif contribution > top_heap[0][0]:
                    heapq.heapreplace(top_heap, entry)

            score = 1.0 - product
            direction = (pol_num / pol_den) if pol_den > 0 else None
            coverage = (pol_den / total_c) if total_c > 0 else 0.0
            top_contributors = [
                AttributionRow(
                    event_id=c.representative_event_id,
                    label_source=c.label_source,
                    event_type=c.event_type,
                    corridor=c.corridor,
                    path=(c.paths or {}).get(s, []),
                    cluster_date=c.cluster_date,
                    contribution=contribution,
                    imp=imp,
                    tau=tau,
                    days_since=days_since,
                    k_value=c.K,
                )
                for contribution, _, c, imp, tau, days_since in sorted(top_heap, reverse=True)
            ]
            results[s].append(DailyScore(
                day=t, flux_score=score, direction=direction,
                direction_coverage=coverage, n_clusters=n_contrib,
                top_contributors=top_contributors,
            ))

    return results


_SCHEMA = """
CREATE TABLE IF NOT EXISTS flux_scores_daily (
    ticker             TEXT NOT NULL,
    date               TEXT NOT NULL,
    flux_score        REAL NOT NULL,
    direction          REAL,
    direction_coverage REAL NOT NULL,
    n_clusters         INTEGER NOT NULL,
    computed_at        TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
);
"""


def store_daily_series(results: dict[str, list], db_path=DEFAULT_DB_PATH) -> int:
    """Idempotent UPSERT (INSERT OR REPLACE, keyed on (ticker, date)) -- safe
    to recompute and re-store after a formula change, matching this project's
    existing storage discipline (ingestion/storage.py, labeling/storage.py)."""
    con = sqlite3.connect(db_path)
    try:
        con.executescript(_SCHEMA)
        computed_at = datetime.now(timezone.utc).isoformat()
        rows = [
            (ticker, d.day.isoformat(), d.flux_score, d.direction,
             d.direction_coverage, d.n_clusters, computed_at)
            for ticker, series in results.items() for d in series
        ]
        con.executemany(
            """
            INSERT OR REPLACE INTO flux_scores_daily
                (ticker, date, flux_score, direction, direction_coverage, n_clusters, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        con.commit()
        return len(rows)
    finally:
        con.close()


_ATTRIBUTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS flux_attribution_daily (
    ticker         TEXT NOT NULL,
    date           TEXT NOT NULL,
    rank           INTEGER NOT NULL,
    event_id       TEXT NOT NULL,
    label_source   TEXT NOT NULL,
    event_type     TEXT NOT NULL,
    corridor       TEXT NOT NULL,
    path           TEXT NOT NULL,
    cluster_date   TEXT NOT NULL,
    contribution   REAL NOT NULL,
    imp            REAL NOT NULL,
    tau            REAL NOT NULL,
    days_since     REAL NOT NULL,
    k_value        REAL NOT NULL,
    computed_at    TEXT NOT NULL,
    PRIMARY KEY (ticker, date, rank)
);
CREATE INDEX IF NOT EXISTS idx_flux_attribution_daily_event ON flux_attribution_daily(event_id);
"""


@dataclass
class EventCatalogEntry:
    """Descriptive fields for one cluster-representative event, Phase I
    (event-centric query). Deliberately does not duplicate corridor/k_value --
    those are already stored per-event in flux_attribution_daily and are
    read from there by api/routers/events.py."""
    event_id: str
    event_type: str
    label_source: str
    severity_score: float
    confidence: float
    polarity: Optional[float]
    countries: list
    published_at: datetime
    url: Optional[str] = None
    title: Optional[str] = None


def build_event_catalog(events: list[dict], clusters: list[Cluster]) -> list[EventCatalogEntry]:
    """One entry per cluster representative, pulled from the already-loaded
    `events` list (the same list load_events() returned to build the clusters
    in the first place) -- no second DB pass over classified_events."""
    by_id = {e["event_id"]: e for e in events}
    entries = []
    seen: set = set()
    for c in clusters:
        if c.representative_event_id in seen:
            continue
        e = by_id.get(c.representative_event_id)
        if e is None:
            continue
        seen.add(c.representative_event_id)
        entries.append(EventCatalogEntry(
            event_id=e["event_id"], event_type=e["event_type"], label_source=e["label_source"],
            severity_score=e["severity_score"], confidence=e["confidence"], polarity=e.get("polarity"),
            countries=e.get("countries", []), published_at=e["published_at"],
        ))
    return entries


def enrich_event_catalog_urls(entries: list[EventCatalogEntry], db_path=DEFAULT_DB_PATH) -> None:
    """One indexed IN-lookup against raw_events for url/title -- the only
    fields load_events() doesn't carry (see flux_engine/query.py). Mutates
    entries in place.

    raw_events.title is only ever populated for rss/newsapi sources -- it's
    always NULL for GDELT (a structured event export with no headline field,
    see ingestion/gdelt.py). For entries still missing a title after the
    raw_events lookup but that do have a url (true for essentially all GDELT
    rows, since GDELT always carries its own SOURCEURL), fall back to the
    scraped-title cache (ingestion/article_titles.py): fetch any URLs not
    already cached, then fill titles from the cache. This is the module's
    only external-network dependency, and it's fully skippable -- a title
    stays None (same as before) if scraping fails or finds nothing."""
    if not entries:
        return
    con = sqlite3.connect(db_path)
    try:
        ids = [e.event_id for e in entries]
        placeholders = ",".join("?" for _ in ids)
        rows = con.execute(
            f"SELECT id, url, title FROM raw_events WHERE id IN ({placeholders})", ids
        ).fetchall()
        by_id = {row[0]: (row[1], row[2]) for row in rows}
        for e in entries:
            url, title = by_id.get(e.event_id, (None, None))
            e.url = url
            e.title = title
    finally:
        con.close()

    missing = [e.url for e in entries if e.title is None and e.url]
    if not missing:
        return
    from src.ingestion.article_titles import fetch_missing_titles, lookup_titles

    fetch_missing_titles(missing, db_path=db_path)
    scraped = lookup_titles(missing, db_path=db_path)
    for e in entries:
        if e.title is None and e.url in scraped:
            e.title = scraped[e.url]


_EVENT_CATALOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS event_catalog (
    event_id       TEXT PRIMARY KEY,
    event_type     TEXT NOT NULL,
    label_source   TEXT NOT NULL,
    severity_score REAL NOT NULL,
    confidence     REAL NOT NULL,
    polarity       REAL,
    countries      TEXT NOT NULL,
    published_at   TEXT NOT NULL,
    url            TEXT,
    title          TEXT,
    computed_at    TEXT NOT NULL
);
"""


def store_event_catalog(
    entries: list[EventCatalogEntry], since: Optional[datetime] = None, db_path=DEFAULT_DB_PATH
) -> int:
    """Full-rebuild-of-a-window write, not a pure upsert. `entries` is the
    complete, authoritative cluster-representative set for events loaded via
    `load_events(since=since)` (see run_timeseries.py) -- so a bucket whose
    winner changed between runs (e.g. after correcting a classified_events
    row -- see labeling/run_gdelt_reclassify.py) means the OLD winner's
    event_id is simply absent from `entries` this time. Plain
    `INSERT OR REPLACE` alone never removes that old row, since it only
    touches PKs present in the new data -- discovered when a rerun after a
    correction pass left 3,429 stale pre-correction rows sitting alongside
    4,863 fresh ones (event_catalog total: 8,292, not 4,863) with no way to
    tell them apart except a stale `computed_at`. Deleting first, scoped to
    `since` (the same cutoff `load_events` used, so this can never delete
    outside what `entries` had authority over), makes this a true window
    rebuild instead of an accumulate-forever upsert."""
    con = sqlite3.connect(db_path)
    try:
        con.executescript(_EVENT_CATALOG_SCHEMA)
        if since is not None:
            con.execute("DELETE FROM event_catalog WHERE published_at >= ?", (since.isoformat(),))
        computed_at = datetime.now(timezone.utc).isoformat()
        rows = [
            (e.event_id, e.event_type, e.label_source, e.severity_score, e.confidence,
             e.polarity, json.dumps(e.countries), e.published_at.isoformat(),
             e.url, e.title, computed_at)
            for e in entries
        ]
        con.executemany(
            """
            INSERT OR REPLACE INTO event_catalog
                (event_id, event_type, label_source, severity_score, confidence,
                 polarity, countries, published_at, url, title, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        con.commit()
        return len(rows)
    finally:
        con.close()


def store_daily_attribution(results: dict[str, list], db_path=DEFAULT_DB_PATH) -> int:
    """Idempotent UPSERT of each DailyScore's top_contributors (Phase H
    event-level drill-down), keyed (ticker, date, rank). Sibling to
    store_daily_series -- same discipline, same call site (see
    flux_engine/run_timeseries.py). Deletes-then-inserts each (ticker,date)
    that appears in `results` so a shrinking top_k on a later rerun can't
    leave stale extra ranks behind (INSERT OR REPLACE alone wouldn't catch
    that, since it only overwrites matching PKs, not removes extras)."""
    con = sqlite3.connect(db_path)
    try:
        con.executescript(_ATTRIBUTION_SCHEMA)
        computed_at = datetime.now(timezone.utc).isoformat()
        rows = []
        seen_days: set[tuple] = set()
        for ticker, series in results.items():
            for d in series:
                if not d.top_contributors:
                    continue
                seen_days.add((ticker, d.day.isoformat()))
                for rank, a in enumerate(d.top_contributors, start=1):
                    rows.append((
                        ticker, d.day.isoformat(), rank,
                        a.event_id, a.label_source, a.event_type,
                        json.dumps(list(a.corridor)), json.dumps(list(a.path)),
                        a.cluster_date.isoformat(), a.contribution, a.imp, a.tau,
                        a.days_since, a.k_value, computed_at,
                    ))
        con.executemany(
            "DELETE FROM flux_attribution_daily WHERE ticker = ? AND date = ?",
            list(seen_days),
        )
        con.executemany(
            """
            INSERT OR REPLACE INTO flux_attribution_daily
                (ticker, date, rank, event_id, label_source, event_type, corridor,
                 path, cluster_date, contribution, imp, tau, days_since, k_value, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        con.commit()
        return len(rows)
    finally:
        con.close()
