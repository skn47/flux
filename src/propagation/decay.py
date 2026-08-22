"""
Propagation decay for Phase D. Two DISTINCT mechanisms (progress/exposure_graph.md §4):

1. Graph-distance decay -- how a flux attenuates over hops. We let the edge
   weights themselves carry the attenuation: impact along a path = product of the
   edge weights. We deliberately DO NOT add an artificial per-hop penalty
   (hop_penalty defaults to 1.0). The weights are already <1 transmission
   fractions, so a 2-hop path (0.9 * 0.75 = 0.675) is already attenuated below the
   1-hop origin (0.9). A separate penalty would double-count with no real-world
   basis. `hop_penalty` is exposed only so a caller can experiment; leaving it 1.0
   is the justified default.

2. Time decay -- how a flux fades in the days after the event, as an exponential
   half-life. The half-life constant is EXPLICITLY A JUDGMENT CALL, NOT CALIBRATED
   against real market data (same honesty standard as Phase B's severity weights,
   progress/event_taxonomy.md §3.6). Default 5 trading days; revisit in Phase G once
   there is outcome data to fit against.
"""

from __future__ import annotations

# Placeholder-but-principled default. NOT fitted to market data -- see module docstring.
DEFAULT_HALF_LIFE_DAYS = 5.0


def graph_distance_impact(edge_weights, hop_penalty: float = 1.0) -> float:
    """Product of edge weights along a path (optionally * hop_penalty**(hops-1);
    hop_penalty=1.0 by default = no extra penalty, see module docstring)."""
    impact = 1.0
    for w in edge_weights:
        impact *= w
    hops = max(0, len(edge_weights) - 1)
    return impact * (hop_penalty ** hops)


def time_decay(days_since_event: float, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    """Exponential half-life multiplier in (0, 1]. 0.5 ** (days / half_life)."""
    if days_since_event < 0:
        raise ValueError("days_since_event must be >= 0")
    if half_life_days <= 0:
        raise ValueError("half_life_days must be > 0")
    return 0.5 ** (days_since_event / half_life_days)


def propagated_impact(base_impact: float, days_since_event: float,
                       half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    """Combine a graph-distance impact with time decay for a given day offset."""
    return base_impact * time_decay(days_since_event, half_life_days)
