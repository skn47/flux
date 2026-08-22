"""
Runnable demonstration of the Phase D exposure graph + decay functions.

Run:  ./.venv/bin/python -m propagation.demo

Runs three concrete example events through the graph and prints per-stock
propagated exposure, then illustrates time decay on one of them. Sanity checks
are printed inline (e.g. a Taiwan event should hit TSM hardest, INTC least).
"""

from __future__ import annotations

from src.propagation.graph import ExposureGraph, NODES, STOCK
from src.propagation.decay import time_decay

SCENARIOS = [
    ("Taiwan earthquake", {"Taiwan"}, "natural_disaster"),
    ("US export-control action on China", {"United States"}, "trade_export_control"),
    ("South Korea fab labor dispute", {"South Korea"}, "supply_chain_fab_disruption"),
]

TICKERS = [n for n, t in NODES.items() if t == STOCK]


def main() -> None:
    graph = ExposureGraph()

    header = f"{'event':<36} " + " ".join(f"{t:>7}" for t in TICKERS)
    print(header)
    print("-" * len(header))
    for label, countries, event_type in SCENARIOS:
        res = graph.query(countries, event_type)
        cells = " ".join(f"{res[t].impact:7.3f}" if t in res else f"{'  -  ':>7}" for t in TICKERS)
        print(f"{label:<36} {cells}")

    print("\nStrongest transmission path per stock (Taiwan earthquake):")
    for exp in sorted(graph.query({"Taiwan"}, "natural_disaster").values(),
                       key=lambda s: s.impact, reverse=True):
        print(f"  {exp.ticker:>5} {exp.impact:5.3f}  via {' -> '.join(exp.path)}")

    print("\nTime decay (NVDA, Taiwan earthquake, half-life 5 trading days):")
    base = graph.query({"Taiwan"}, "natural_disaster")["NVDA"].impact
    for d in (0, 3, 5, 10):
        print(f"  day {d:>2}: {base * time_decay(d):5.3f}")


if __name__ == "__main__":
    main()
