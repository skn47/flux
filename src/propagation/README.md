# Propagation (Phase D)

Country <-> sector <-> stock exposure graph + flux-propagation decay functions.
Consumed by the Phase E flux-scoring formula: given a classified event's affected
country/countries and event_type, it returns which tracked stocks are exposed and
by how much. Every edge weight is grounded in real, cited evidence or an explicit
flagged judgment call -- full justification in `progress/exposure_graph.md`. No
hash-derived / placeholder weights (the Nexus anti-pattern; see
`progress/nexus_comparison.md`).

## Schema
- **Nodes** (9): 3 country (United States, Taiwan, South Korea), 1 sector
  (semiconductors), 5 stock (NVDA, TSM, AMD, ASML, INTC).
- **Edges** (31, directed, weight in [0,1]):
  - country -> stock, tagged with a `channel` (manufacturing | demand | policy |
    supply) describing the exposure kind.
  - stock -> stock supply-chain (structural transmission, channel None).
  - country -> sector (channel sector_broad) and sector -> stock (transmission),
    for broad / non-geographically-specific events.
- Each edge's `note` carries a `§anchor` into `progress/exposure_graph.md`.

## Querying
```python
from propagation.graph import ExposureGraph
g = ExposureGraph()
result = g.query({"Taiwan"}, "natural_disaster")   # {ticker: StockExposure}
for exp in sorted(result.values(), key=lambda s: s.impact, reverse=True):
    print(exp.ticker, round(exp.impact, 3), exp.path)
```
`event_type` must be one of the Phase B taxonomy values (`progress/event_taxonomy.md`).
Impact along a path = `activation(first-hop channel) * product(edge weights)`;
multiple paths to a stock are combined by **max**.

## Decay functions (`propagation/decay.py`)
- **Graph-distance decay** = product of edge weights along a path. No artificial
  per-hop penalty (`hop_penalty=1.0`): weights are already <1, so multi-hop paths
  are attenuated by construction.
- **Time decay** = exponential half-life, `0.5 ** (days / half_life)`. Half-life
  default **5 trading days is an EXPLICIT, UNCALIBRATED judgment call** (not fitted
  to market data) -- revisit in Phase G, matching Phase B's honesty standard.

## Demo
```
./.venv/bin/python -m propagation.demo
```
Runs a Taiwan earthquake, a US export-control action, and a South Korea fab labor
dispute; prints per-stock exposure, the strongest path per stock, and a time-decay
illustration.

## Known limitations
1. **China is not a node** but is the dominant export-control transmission channel;
   proxied via `US->stock [policy]` weights calibrated to observed export-control
   sensitivity. Biggest structural gap (add China in Phase E/F).
2. **ASML HQ (NL) + largest market (China) are outside the corridor** -> its
   absolute risk is understated.
3. **Revenue-geography disclosures are billed-to, not end-demand** (esp. NVDA);
   demand weights are flagged judgment calls.
4. **Channel-activation table + sector betas are uncalibrated judgment calls.**
5. **Static hand-curated 2024/25 snapshot** -- no auto-refresh; review vs. new filings.
6. **Single sector (semiconductors)** per MVP -- sector node is a passthrough today.
7. Modeling choices (max-over-paths, activation-at-first-hop, no hop penalty) are
   documented in `progress/exposure_graph.md` §4, not the only defensible options.
