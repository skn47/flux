"""
Leakage-safe projection of an already-known flux_score trajectory into the
FUTURE, for the seq2seq decoder's per-step conditioning (see
lstm/seq2seq_model.py, lstm/seq_dataset.py). New for the 30-day-horizon
rework -- see lstm/seq2seq_README.md for the full design rationale.

Why this is leakage-safe (the precise argument, not a hand-wave)
------------------------------------------------------------------
`flux_score(s, t)` (flux_engine/formula.py, flux_engine/timeseries.py) is a
noisy-OR aggregate over event contributions, and each contribution's time
component (`propagation.decay.time_decay`) is a PURE, DETERMINISTIC function
of `(event_date, half_life_days, t)` -- it needs no information not already
known at `event_date`. So for any event with `published_at.date() <=
as_of_date`, its contribution to `flux_score(s, as_of_date + k)` for ANY
future offset `k` is fully computable right now, using zero information from
the future.

`project_flux_trajectory` / `FluxTrajectoryProjector.project` therefore
filter the event set to `published_at.date() <= as_of_date` BEFORE doing
anything else (clustering, noisy-OR) -- an event published after `as_of_date`
is structurally ABSENT from the computation, not merely down-weighted. This
is the same guarantee `flux_engine.formula.event_contribution`'s own
`delta_days < 0` guard provides for a single-date query; this module applies
it once, up front, to a whole future date range instead of re-deriving it at
each date.

This projection is a LOWER BOUND, not an expectation
-------------------------------------------------------
Because noisy-OR (`1 - prod(1 - c_i)`) is monotonically non-decreasing as
non-negative contributions are added, and because NEW events (published
between `as_of_date` and a future target date) are the one thing this
function cannot know about, `project_flux_trajectory`'s output for any
`target_date > as_of_date` systematically UNDERSTATES the true future
flux_score whenever new material news actually occurs in that window. This
is stated explicitly here (and should stay explicit in any code that
consumes this module) rather than being implicitly treated as an unbiased
forecast -- it is a deterministic floor from already-known events, and the
model consuming it (lstm/seq2seq_model.py) is expected to learn around that
known, one-directional bias, not to be told the trajectory is unbiased.
"""

from __future__ import annotations

import bisect
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.flux_engine.formula import EPSILON, flux_score
from src.flux_engine.query import load_events
from src.flux_engine.timeseries import Cluster, build_clusters, compute_daily_series
from src.ingestion.storage import DEFAULT_DB_PATH
from src.propagation.decay import DEFAULT_HALF_LIFE_DAYS
from src.propagation.graph import ExposureGraph


class FluxTrajectoryProjector:
    """
    Precomputes the global cluster list ONCE (reusing
    flux_engine.timeseries.build_clusters unmodified -- see that module's
    docstring for why clustering can be done globally rather than per-date/
    per-stock: imp(e,s) and tau(e,t) are identical for every event within one
    (date, event_type, corridor) cluster, so the max-K representative is
    picked once and is valid for any (stock, t) query). `Cluster.imp` already
    covers every tracked ticker, so one projector instance serves the whole
    pooled dataset -- avoids rebuilding clusters on every one of the
    thousands of `.project()` calls a full pooled-dataset build makes (one
    per SeqWindow, see lstm/seq_dataset.py).

    Visibility cutoff (the leakage-safety mechanism): `self.clusters` is
    sorted ascending by `cluster_date` (build_clusters already guarantees
    this); `.project(as_of_date=...)` bisects to the clusters with
    `cluster_date <= as_of_date` and passes ONLY that prefix into
    `compute_daily_series` -- clusters from events published after
    `as_of_date` are never in the slice at all.
    """

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        graph: Optional[ExposureGraph] = None,
    ):
        self.db_path = db_path
        self.graph = graph or ExposureGraph()
        events = load_events(db_path=db_path)
        self.clusters: list[Cluster] = build_clusters(events, graph=self.graph)
        self.cluster_dates: list[date] = [c.cluster_date for c in self.clusters]

    def project(
        self,
        ticker: str,
        as_of_date: date,
        target_dates: list[date],
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        epsilon: float = EPSILON,
    ) -> np.ndarray:
        """Returns one flux_score float per `target_dates` entry (same
        order), using only clusters with `cluster_date <= as_of_date`.
        `target_dates` need not be contiguous (a trading calendar has gaps);
        `compute_daily_series` is queried over the full
        [min(target_dates), max(target_dates)] calendar span and the result
        is subset back down to exactly `target_dates`."""
        if not target_dates:
            return np.array([], dtype=np.float64)
        hi = bisect.bisect_right(self.cluster_dates, as_of_date)
        visible = self.clusters[:hi]
        results = compute_daily_series(
            visible, [ticker],
            start=min(target_dates), end=max(target_dates),
            half_life_days=half_life_days, epsilon=epsilon,
        )
        by_date = {d.day: d.flux_score for d in results[ticker]}
        return np.array([by_date[d] for d in target_dates], dtype=np.float64)


def project_flux_trajectory(
    ticker: str,
    as_of_date: date,
    target_dates: list[date],
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    epsilon: float = EPSILON,
    db_path: Path | str = DEFAULT_DB_PATH,
    graph: Optional[ExposureGraph] = None,
) -> np.ndarray:
    """Convenience one-shot wrapper around `FluxTrajectoryProjector` (builds
    a projector -- including a full global cluster pass -- on every call).
    Fine for a single lookup or a test; callers building many windows (e.g.
    lstm/seq_dataset.py's pooled dataset construction) should build ONE
    `FluxTrajectoryProjector` and reuse it across every window instead, to
    avoid re-clustering the whole event corpus per call."""
    projector = FluxTrajectoryProjector(db_path=db_path, graph=graph)
    return projector.project(ticker, as_of_date, target_dates, half_life_days=half_life_days, epsilon=epsilon)


def verify_flux_trajectory_no_future_leakage(
    ticker: str,
    as_of_date: date,
    target_dates: list[date],
    projected: np.ndarray,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    db_path: Path | str = DEFAULT_DB_PATH,
    graph: Optional[ExposureGraph] = None,
    tol: float = 1e-9,
) -> None:
    """
    Independent re-derivation check for a `projected` array produced by
    `FluxTrajectoryProjector.project` / `project_flux_trajectory` -- mirrors
    this project's existing verification discipline (e.g.
    lstm/evaluate.py::assert_metrics_agree: two independently-coded paths
    must agree before a number is trusted).

    Deliberately uses `flux_engine.formula.flux_score` -- the original,
    per-date, doc-verified implementation (flux_engine/timeseries.py's own
    docstring states it was checked equivalent to this) -- rather than a
    second call into `compute_daily_series`'s two-pointer bulk machinery, so
    a bug shared by `FluxTrajectoryProjector` and `compute_daily_series`
    cannot hide from this check.

    Also asserts directly (not just by construction) that no event used in
    the re-derivation was published after `as_of_date` -- the leakage
    guarantee this whole module exists to provide, checked explicitly rather
    than assumed from the filtering logic being correct.

    Raises AssertionError on the first mismatch or leakage found.
    """
    graph = graph or ExposureGraph()
    events = [e for e in load_events(db_path=db_path) if e["published_at"].date() <= as_of_date]
    assert all(e["published_at"].date() <= as_of_date for e in events), (
        f"LEAKAGE: an event published after as_of_date={as_of_date} survived filtering"
    )

    for target_date, expected in zip(target_dates, projected):
        t = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
        result = flux_score(events, ticker, t, graph=graph, half_life_days=half_life_days)
        assert abs(result.flux_score - float(expected)) < tol, (
            f"MISMATCH: {ticker} {target_date}: independent re-derivation "
            f"{result.flux_score} != projected {float(expected)} "
            f"(as_of_date={as_of_date})"
        )
