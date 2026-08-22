"""
Country <-> sector <-> stock exposure graph for Phase D.

An explicit, hand-curated weighted directed graph. Every edge weight is traceable
to a justification in progress/exposure_graph.md via a `§anchor` in its `note` field --
NEVER a hash-derived or arbitrary placeholder (the specific anti-pattern called out
in progress/nexus_comparison.md, where Nexus's supplyChainLinkage == hash(ticker) % N).

Design choice: a plain adjacency dict, not networkx. The graph is small (32 nodes, 112 edges
as of the S3 pharma_biotech-sector addition -- see progress/sector_expansion_pharma.md; counts
drift as sectors are added, always independently recountable via len(NODES)/len(EDGES) rather
than trusted from this comment), the traversal we need is a bounded max-product DFS, and
avoiding a new dependency keeps this auditable end-to-end. Documented here so the choice
is explicit.

Node types: country | sector | stock.
Edge channels: manufacturing | demand | policy | supply (country->stock exposure
kinds), sector_broad (country->sector), or None (pure structural transmission on
stock->stock and sector->stock edges).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --- node types -----------------------------------------------------------
COUNTRY = "country"
SECTOR = "sector"
STOCK = "stock"

SECTOR_NODE = "semiconductors"  # this sector's node name -- see progress/exposure_graph.md
FINANCIALS_SECTOR_NODE = "financials"  # S1 (2026-07-23) -- see progress/sector_expansion_financials.md
ENERGY_SECTOR_NODE = "energy"  # S2 (2026-07-23) -- see progress/sector_expansion_energy.md
PHARMA_SECTOR_NODE = "pharma_biotech"  # S3 (2026-07-23) -- see progress/sector_expansion_pharma.md
# S0 REWORK (2026-07-23): the graph mechanics below (Edge, NODES, EDGES,
# ExposureGraph's DFS traversal) were already sector-agnostic -- they just walk
# whatever nodes/edges exist. Adding a second sector (S1/S2/S3: financials,
# energy, pharma_biotech, see the plan's "Sector & ticker broadening" status
# update) means adding that sector's own SECTOR-typed node (e.g. "financials")
# plus its own country->sector and sector->stock edges below -- a pure data
# addition, not a structural change to this module. `config.mvp_scope.SECTORS`
# is the registry of which sectors exist; this file's NODES/EDGES independently
# encode each one's actual exposure relationships (the graph is hand-curated
# and cited, not derived from the config registry).
#
# S1 note: Financials' country->stock edges use only the DEMAND and POLICY
# channels (never MANUFACTURING/SUPPLY, which describe physical fab/wafer
# supply chains that have no analogue for a domestic banking/insurance/
# brokerage sector) and carry NO stock->stock edges at all -- researched and
# confirmed no real, citable, material bilateral counterparty/underwriting
# relationship exists among the 5 financials tickers (see
# progress/sector_expansion_financials.md §2/§3), an explicit documented decision
# rather than an oversight, per the plan's own "it's fine if financials
# tickers have NO stock->stock edges" allowance.
#
# S2 note: Energy is the first sector needing country nodes OUTSIDE the original
# US/Taiwan/South Korea corridor -- Saudi Arabia and Russia, verified against
# GDELT's own live reference files (see progress/sector_expansion_energy.md §2a and
# config/mvp_scope.py's energy sector entry; Russia's FIPS code is "RS", NOT
# ISO's "RU" -- a second real example of the same class of trap South Korea's
# "KS"-not-"KR" already demonstrated). `Saudi Arabia` is ALSO used, explicitly
# flagged with a ⚠, as a documented proxy node for OXY's actual Oman/UAE/Qatar/
# Algeria Gulf-region operations (those countries are not separately registered,
# per the scope decision in progress/sector_expansion_energy.md §2 not to multiply
# corridor-verification work past Saudi Arabia's own swing-producer role) -- the
# same kind of documented proxy-node judgment call this file's China-is-not-a-
# node limitation (progress/exposure_graph.md §5 limitation 1) already uses for the
# semiconductor sector. Energy also has NO stock->stock edges, for a reason
# distinct from financials': researched and found no NAMED bilateral supply/
# customer relationship comparable to TSM->NVDA's disclosed foundry-customer
# link -- crude oil/refined-product/LNG flows between these 5 tickers are
# fungible commodity-market transactions, not named bilateral contracts, per
# progress/sector_expansion_energy.md §7.
#
# S3 note: Pharma/biotech is US-only, FDA-driven -- reuses the exact existing
# "United States" node verbatim (a deliberate scope decision made BEFORE S3
# started, to avoid retriggering the broad cross-sector flux_score shift the
# Saudi Arabia/Russia relabel discovered is caused by ANY new country node --
# see progress/sector_expansion_pharma.md and the plan's own "Saudi Arabia/Russia
# relabel" finding). Like financials, pharma_biotech's country->stock edges use
# only DEMAND/POLICY channels (never MANUFACTURING/SUPPLY -- a single-country
# corridor has no country-to-country manufacturing/supply-geography contrast to
# model, and no citation basis was researched for one) and carry NO stock->stock
# edges -- researched and confirmed no real, citable, NAMED bilateral
# relationship exists among the 5 tracked tickers themselves (their real
# collaboration/partnership structures run through non-tracked companies: e.g.
# Regeneron/Sanofi, Pfizer/BioNTech, Vertex/CRISPR Therapeutics -- see
# progress/sector_expansion_pharma.md §3).

# --- exposure channels ----------------------------------------------------
MANUFACTURING = "manufacturing"
DEMAND = "demand"
POLICY = "policy"
SUPPLY = "supply"
SECTOR_BROAD = "sector_broad"
TRANSMIT = None  # structural transmission edges carry no activation channel


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    weight: float          # transmission fraction in [0, 1]
    channel: Optional[str]
    note: str              # §anchor into progress/exposure_graph.md


# --- nodes ------------------------------------------------------------------
NODES: dict[str, str] = {
    "United States": COUNTRY,
    "Taiwan": COUNTRY,
    "South Korea": COUNTRY,
    SECTOR_NODE: SECTOR,
    "NVDA": STOCK,
    "TSM": STOCK,
    "AMD": STOCK,
    "ASML": STOCK,
    "INTC": STOCK,
    # --- R3 (2026-07-23) additions -- see progress/ticker_universe_expansion.md ---
    "MU": STOCK,
    "QCOM": STOCK,
    "AMAT": STOCK,
    # --- S1 (2026-07-23) additions -- see progress/sector_expansion_financials.md ---
    FINANCIALS_SECTOR_NODE: SECTOR,
    "JPM": STOCK,
    "SCHW": STOCK,
    "ZION": STOCK,
    "PGR": STOCK,
    "COF": STOCK,
    # --- S2 (2026-07-23) additions -- see progress/sector_expansion_energy.md ---
    "Saudi Arabia": COUNTRY,
    "Russia": COUNTRY,
    ENERGY_SECTOR_NODE: SECTOR,
    "XOM": STOCK,
    "OXY": STOCK,
    "EOG": STOCK,
    "LNG": STOCK,
    "VLO": STOCK,
    # --- S3 (2026-07-23) additions -- see progress/sector_expansion_pharma.md ---
    # No new country node -- reuses "United States" verbatim (see note above).
    PHARMA_SECTOR_NODE: SECTOR,
    "PFE": STOCK,
    "VRTX": STOCK,
    "BMRN": STOCK,
    "MRNA": STOCK,
    "REGN": STOCK,
}

# --- edges (see progress/exposure_graph.md §2 for every weight's justification) ---
EDGES: list[Edge] = [
    # 2a. country -> stock
    Edge("Taiwan", "TSM", 0.90, MANUFACTURING, "§TSM-mfg: >90% capacity in Taiwan"),
    Edge("United States", "TSM", 0.10, MANUFACTURING, "§TSM-mfg-us: Arizona <10%"),
    Edge("United States", "TSM", 0.70, DEMAND, "§TSM-dem: US ~70%+ revenue"),
    Edge("United States", "TSM", 0.45, POLICY, "§TSM-pol: US export-rule compliance"),
    Edge("United States", "NVDA", 0.60, POLICY, "§NVDA-pol: China GPU bans, disclosed $B"),
    Edge("United States", "NVDA", 0.45, DEMAND, "§NVDA-dem: judgment (billed-to != end-demand)"),
    Edge("South Korea", "NVDA", 0.25, SUPPLY, "§NVDA-kr: HBM from SK Hynix/Samsung"),
    Edge("United States", "AMD", 0.45, POLICY, "§AMD-pol: MI-series China restrictions"),
    Edge("United States", "AMD", 0.40, DEMAND, "§AMD-dem: judgment, below NVDA"),
    Edge("South Korea", "AMD", 0.20, SUPPLY, "§AMD-kr: HBM, smaller base"),
    Edge("Taiwan", "ASML", 0.13, DEMAND, "§ASML-tw: Taiwan ~11% of system sales"),
    Edge("South Korea", "ASML", 0.28, DEMAND, "§ASML-kr: Korea ~28% of system sales"),
    Edge("United States", "ASML", 0.50, POLICY, "§ASML-pol: US-led export controls on China sales"),
    Edge("United States", "INTC", 0.70, MANUFACTURING, "§INTC-mfg: majority US fabs"),
    Edge("Taiwan", "INTC", 0.20, MANUFACTURING, "§INTC-tw: TSMC-outsourced tiles"),
    Edge("United States", "INTC", 0.45, DEMAND, "§INTC-dem: judgment, large US revenue"),
    Edge("United States", "INTC", 0.30, POLICY, "§INTC-pol: CHIPS often favorable -> lowest"),

    # --- R3 (2026-07-23) additions: MU, QCOM, AMAT -- see progress/ticker_universe_expansion.md ---
    Edge("Taiwan", "MU", 0.45, MANUFACTURING, "§MU-mfg: Taichung largest DRAM hub + Tongluo P5, no exact % disclosed"),
    Edge("United States", "MU", 0.35, MANUFACTURING, "§MU-mfg-us: Boise/Manassas + new ID/NY fabs, ~40% domestic DRAM goal"),
    Edge("United States", "MU", 0.60, DEMAND, "§MU-dem: US 64.51% of FY2025 revenue by geography (cited; billed-to ambiguity)"),
    Edge("South Korea", "MU", 0.30, SUPPLY, "§MU-kr: judgment -- competitive memory-market exposure (Samsung/SK Hynix HBM share), not a sourcing dependency; MU has no Korea fab"),
    Edge("United States", "MU", 0.55, POLICY, "§MU-pol: China CAC 2023 ban of MU from critical infrastructure; China rev $3.05B->$2.64B FY24->FY25"),

    Edge("South Korea", "QCOM", 0.55, DEMAND, "§QCOM-kr: Korea 20% of FY2024 revenue by customer HQ (cited; Samsung a >=10% customer)"),
    Edge("United States", "QCOM", 0.25, DEMAND, "§QCOM-dem: US 25% of FY2024 revenue by customer HQ (cited)"),
    Edge("United States", "QCOM", 0.30, POLICY, "§QCOM-pol: judgment -- moderate export-control sensitivity, below NVDA"),

    Edge("Taiwan", "AMAT", 0.24, DEMAND, "§AMAT-tw: Taiwan 24% of FY2025 net revenue (cited)"),
    Edge("South Korea", "AMAT", 0.20, DEMAND, "§AMAT-kr: Korea 20% of FY2025 net revenue (cited)"),
    Edge("United States", "AMAT", 0.45, POLICY, "§AMAT-pol: China export-curb impact, China share 45%(FY24 peak)->25%(Q4 FY25), still largest single market"),

    # 2b. stock -> stock supply-chain (structural transmission; channel=None)
    Edge("TSM", "NVDA", 0.75, TRANSMIT, "§sc-tsm-nvda: NVDA ~100% leading-edge from TSMC"),
    Edge("TSM", "AMD", 0.70, TRANSMIT, "§sc-tsm-amd: AMD leading-edge from TSMC, some GF"),
    Edge("TSM", "INTC", 0.25, TRANSMIT, "§sc-tsm-intc: Intel outsources some tiles"),
    Edge("TSM", "ASML", 0.40, TRANSMIT, "§sc-tsm-asml: TSMC is ASML's largest customer"),
    Edge("ASML", "TSM", 0.35, TRANSMIT, "§sc-asml-tsm: TSMC advanced nodes need ASML EUV"),
    Edge("ASML", "INTC", 0.30, TRANSMIT, "§sc-asml-intc: Intel 18A/High-NA needs ASML"),

    # --- R3 (2026-07-23) additions: MU, QCOM, AMAT connections to existing 5 ---
    Edge("TSM", "QCOM", 0.65, TRANSMIT, "§sc-tsm-qcom: TSMC manufactures majority of flagship Snapdragon chips"),
    Edge("TSM", "AMAT", 0.35, TRANSMIT, "§sc-tsm-amat: TSMC is/was AMAT's single largest customer"),
    Edge("AMAT", "TSM", 0.30, TRANSMIT, "§sc-amat-tsm: judgment -- TSMC fabs depend on AMAT wafer-fab equipment (mirrors existing bidirectional ASML<->TSM pattern)"),
    Edge("AMAT", "INTC", 0.30, TRANSMIT, "§sc-amat-intc: AMAT disclosed equipment supplier to Intel's fabs"),
    Edge("ASML", "MU", 0.20, TRANSMIT, "§sc-asml-mu: MU adopting ASML EUV for 1-gamma DRAM (2025), but last memory maker to join EUV, limited/reduced use so far"),

    # 2c. sector edges
    Edge("Taiwan", SECTOR_NODE, 0.90, SECTOR_BROAD, "§sec-tw"),
    Edge("United States", SECTOR_NODE, 0.80, SECTOR_BROAD, "§sec-us"),
    Edge("South Korea", SECTOR_NODE, 0.55, SECTOR_BROAD, "§sec-kr"),
    Edge(SECTOR_NODE, "NVDA", 0.80, TRANSMIT, "§sec-nvda: sector beta"),
    Edge(SECTOR_NODE, "TSM", 0.90, TRANSMIT, "§sec-tsm: sector beta"),
    Edge(SECTOR_NODE, "AMD", 0.80, TRANSMIT, "§sec-amd: sector beta"),
    Edge(SECTOR_NODE, "ASML", 0.70, TRANSMIT, "§sec-asml: sector beta"),
    Edge(SECTOR_NODE, "INTC", 0.70, TRANSMIT, "§sec-intc: sector beta"),
    # --- R3 (2026-07-23) additions ---
    Edge(SECTOR_NODE, "MU", 0.85, TRANSMIT, "§sec-mu: memory sector beta, judgment (arguably more cyclical than logic)"),
    Edge(SECTOR_NODE, "QCOM", 0.75, TRANSMIT, "§sec-qcom: sector beta, judgment"),
    Edge(SECTOR_NODE, "AMAT", 0.65, TRANSMIT, "§sec-amat: equipment maker, judgment (more diversified across memory+logic+China than ASML)"),

    # --- S1 (2026-07-23) additions: financials sector -- see progress/sector_expansion_financials.md ---
    # 2d. country -> stock (financials; DEMAND/POLICY only, no manufacturing/supply channel applies)
    #
    # ⚠ Tie-break note (found during S1 verification, see progress/sector_expansion_financials.md
    # §6): an earlier draft of these weights gave ZION and COF numerically IDENTICAL
    # (demand, policy, sector-transmit) triples, which made them graph-topologically
    # indistinguishable -- same imp(e,s) for every event_type, every day, defeating this
    # sector's own criterion-3 diversity goal even though the two tickers' underlying
    # citations are genuinely different mechanisms. Independently of that, SCHW's and
    # ZION's weights coincidentally produced the exact same geopolitical_military_tension
    # impact (0.8*0.3 == 0.6*0.4 == 0.24) via two DIFFERENT channels -- a real numeric
    # coincidence, not a duplication bug, but one that (combined with the ZION/COF
    # duplication and this corridor's pre-existing event corpus being ~99.5%
    # geopolitical_military_tension-typed, see progress/sector_expansion_financials.md §6)
    # made 3 of the 5 financials tickers' daily flux_score byte-identical on 725/735 days.
    # Weights below are nudged (within the same already-⚠-flagged judgment-call range, same
    # citations) specifically to avoid exact cross-ticker numeric ties, verified by an
    # exhaustive check across all 8 event_types x all 5 tickers (see docs, §6) -- not
    # re-fitted to any backtest/return outcome.
    Edge("United States", "JPM", 0.75, DEMAND, "§JPM-dem: North America 76.56% of FY2025 total net revenue, 10-K footnote 'substantially reflects the U.S.' (cited)"),
    Edge("United States", "JPM", 0.65, POLICY, "§JPM-pol: largest US G-SIB, highest capital surcharge tier, Basel III Endgame exposure (cited)"),
    Edge("United States", "SCHW", 0.80, DEMAND, "§SCHW-dem: no disclosed material international revenue; ASC 280 geographic segment note absent (immateriality inference). ⚠ Judgment, no exact % unlike JPM."),
    Edge("United States", "SCHW", 0.45, POLICY, "§SCHW-pol: Schwab Bank subject to bank-regulatory capital/liquidity oversight + SEC/FINRA broker-dealer regulation; below JPM's G-SIB apparatus. ⚠ Judgment."),
    Edge("United States", "ZION", 0.68, DEMAND, "§ZION-dem: 100% domestic revenue but geographically narrower (11 western states) than JPM's nationwide book (cited footprint). ⚠ Judgment on magnitude."),
    Edge("United States", "ZION", 0.62, POLICY, "§ZION-pol: FDIC special assessment (SVB/Signature depositor protection) + proximity to the $100B enhanced-prudential-standards asset threshold (~$88-89B total assets) -- real, ticker-specific, dated (cited)."),
    Edge("United States", "PGR", 0.60, DEMAND, "§PGR-dem: 'operates throughout the United States' (10-K, cited), no international operations disclosed."),
    Edge("United States", "PGR", 0.50, POLICY, "§PGR-pol: state-by-state insurance rate-approval regulation (cited CA/TX 2023 rate-hike examples) -- a structurally different POLICY mechanism (state insurance regulators) from the bank-capital-rule POLICY edges above."),
    Edge("United States", "COF", 0.72, DEMAND, "§COF-dem: Domestic Card >90% of Credit Card segment net revenue; Consumer/Commercial Banking segments fully domestic (cited)."),
    Edge("United States", "COF", 0.57, POLICY, "§COF-pol: Discover Financial Services acquisition required Fed/OCC approval with remediation conditions (announced 2024-02-19, approved 2025-04-18, closed 2025-05-18, cited) + CFPB card-practice scrutiny."),

    # 2e. sector edges (financials)
    Edge("United States", FINANCIALS_SECTOR_NODE, 0.75, SECTOR_BROAD, "§sec-fin-us: broad US bank-regulatory/macro environment relevance to the financials sector. Judgment."),
    Edge(FINANCIALS_SECTOR_NODE, "JPM", 0.85, TRANSMIT, "§sec-jpm: largest, most representative name, highest sector beta. Judgment."),
    Edge(FINANCIALS_SECTOR_NODE, "SCHW", 0.65, TRANSMIT, "§sec-schw: rate/client-cash-driven business model, more idiosyncratic than a generic bank -- slightly decoupled. ⚠ Judgment."),
    Edge(FINANCIALS_SECTOR_NODE, "ZION", 0.78, TRANSMIT, "§sec-zion: regional banks trade as a real correlated sub-basket, especially during sector-wide stress (e.g. March 2023). Judgment."),
    Edge(FINANCIALS_SECTOR_NODE, "PGR", 0.50, TRANSMIT, "§sec-pgr: underwriting-driven earnings cycle is a genuinely different economic driver than banking -- most decoupled of the 5. ⚠ Judgment."),
    Edge(FINANCIALS_SECTOR_NODE, "COF", 0.70, TRANSMIT, "§sec-cof: large consumer-credit name, correlated with broader bank-sector sentiment. Judgment."),

    # --- S2 (2026-07-23) additions: energy sector -- see progress/sector_expansion_energy.md ---
    # 2f. country -> stock (energy)
    Edge("United States", "XOM", 0.41, DEMAND, "§XOM-dem: US 40.87% of FY2024 total revenue ($138,657M/$339,247M), 10-K geographic revenue table (cited)"),
    Edge("United States", "XOM", 0.60, MANUFACTURING, "§XOM-mfg: US 60.7% of long-lived assets by geography ($178,633M/$294,318M FY2024, 10-K geographic note, cited) -- refining/chemical/upstream PP&E footprint. Found during verification when re-checking already-fetched 10-K text against the same PP&E-geography citation basis used for OXY's MANUFACTURING edge; not fitted to any score/return outcome."),
    Edge("Saudi Arabia", "XOM", 0.30, MANUFACTURING, "§XOM-sa-mfg: 50%-owned Al-Jubail Petrochemical Company JV + Yanbu refining/chemical facilities (cited, 10-K subsidiary table). Judgment on magnitude (minority-JV scale)."),
    Edge("Saudi Arabia", "XOM", 0.55, POLICY, "§XOM-sa-pol: 10-K risk factor names OPEC/OPEC+ production-quota adherence as a supply-affecting factor (cited). Judgment on magnitude."),
    Edge("Russia", "XOM", 0.50, POLICY, "§XOM-rus-pol: Sakhalin-1 exit, $3.0B Q1-2022 + $1.6B + $0.3B 2022 after-tax impairments, dedicated 10-K 'Note 2. Russia' (cited). Judgment on magnitude."),
    Edge("Russia", "XOM", 0.15, SUPPLY, "§XOM-rus-sup: ongoing 7.5% interest in the Caspian Pipeline Consortium, which transits Russia to Black Sea tanker facilities (cited). Deliberately low -- minority interest."),

    Edge("United States", "OXY", 0.90, MANUFACTURING, "§OXY-mfg: US 90.2% of PP&E net by geography ($62,604M/$69,378M FY2024, cited, Permian/DJ Basin core)"),
    Edge("United States", "OXY", 0.20, DEMAND, "§OXY-dem: judgment -- low priority vs. MANUFACTURING for a production-driven E&P, no comparably precise revenue-geography figure found"),
    Edge("Saudi Arabia", "OXY", 0.35, MANUFACTURING, "§OXY-sa-mfg: ⚠ PROXY -- OXY's actual Middle East operations are in Oman/UAE/Qatar/Algeria (largest independent Oman producer; international PP&E $6,774M/$69,378M=9.8%; 25.6% of workforce in the Middle East region, all cited), not Saudi Arabia itself. Modeled on the Saudi Arabia node as the best available proxy for Gulf-region OPEC+-producing-country risk since those countries are not separately registered (see progress/sector_expansion_energy.md §2), same class of documented proxy as the China-via-United-States limitation already flagged for the semiconductor sector."),
    Edge("Saudi Arabia", "OXY", 0.30, POLICY, "§OXY-sa-pol: ⚠ PROXY (see above) -- host-government production-sharing/JV terms (Sonatrach JV in Algeria, ADNOC partnership in UAE, cited entities), modeled on Saudi Arabia for the same reason."),

    Edge("United States", "EOG", 0.95, MANUFACTURING, "§EOG-mfg: 99% of net proved reserves in the United States (1% Trinidad), 10-K cited"),
    Edge("United States", "EOG", 0.30, DEMAND, "§EOG-dem: judgment -- US wellhead/domestic market, no precise revenue-geography % disclosed distinct from the reserves figure"),
    Edge("Saudi Arabia", "EOG", 0.40, POLICY, "§EOG-sa-pol: ⚠ Judgment -- EOG has NO direct Saudi Arabia operations (deliberately the 'pure price-taker' pick, see progress/sector_expansion_energy.md §4); this edge models the global-oil-price transmission effect of OPEC+ production-quota decisions on EOG's realized US wellhead pricing, the only channel by which Saudi Arabia/OPEC+ affects a purely domestic producer."),
    Edge("Russia", "EOG", 0.25, POLICY, "§EOG-rus-pol: ⚠ Judgment -- same price-transmission mechanism as §EOG-sa-pol, smaller weight since Russia's swing-producer/price-setting role is smaller and more sanctions-contingent than Saudi Arabia/OPEC+'s (per Valero's own 10-K, which names both together as crude-oil-differential risk factors)."),

    Edge("United States", "LNG", 0.85, MANUFACTURING, "§LNG-mfg: Sabine Pass + Corpus Christi liquefaction terminals, both on the US Gulf Coast (cited, ~effectively all physical LNG production capacity)"),
    Edge("United States", "LNG", 0.20, POLICY, "§LNG-pol: judgment -- US DOE LNG-export-authorization/permitting regulatory risk, real but not the dominant disclosed risk factor"),
    Edge("Russia", "LNG", 0.55, DEMAND, "§LNG-rus-dem: ⚠ DEMAND-SUBSTITUTION mechanism, structurally different from every other energy edge in this graph -- Cheniere has no Russian operations; its own 10-K states Europe plans '75 mtpa of import capacity... to displace Russian gas imports' and flags the 2024-12-31 expiry of the Russia-Ukraine gas transit agreement as likely to increase European LNG demand (both cited). Russia here drives Cheniere's DEMAND upward via substitution, the opposite sign-mechanism from XOM/VLO's Russia edges (which model Russia as an operational/sourcing risk)."),

    Edge("United States", "VLO", 0.84, MANUFACTURING, "§VLO-mfg: 10 of 15 refineries in the US, 84.2% of throughput capacity (2,685,000/3,190,000 BPD, 10-K refinery-location table, cited)"),
    Edge("United States", "VLO", 0.30, DEMAND, "§VLO-dem: judgment -- US refined-product sales, no separately disclosed revenue-geography % distinct from the throughput-capacity figure"),
    Edge("Saudi Arabia", "VLO", 0.30, SUPPLY, "§VLO-sa-sup: ⚠ Judgment -- 10-K names 'potential sanction adjustments related to Iran, Russia, and Venezuela, the Russia-Ukraine conflict' as a source of crude-oil-differential volatility (cited); Saudi Arabia/OPEC+ swing production is the dominant driver of the global crude differentials this risk factor describes, modeled here as VLO's crude-input-sourcing channel."),
    Edge("Russia", "VLO", 0.45, SUPPLY, "§VLO-rus-sup: 10-K cited -- 'risks attendant to doing business with suppliers... including... the Russia-Ukraine conflict and turmoil in the Middle East'; direct crude-input-sourcing constraint."),
    Edge("Russia", "VLO", 0.35, POLICY, "§VLO-rus-pol: 10-K cited, verbatim -- 'U.S. sanctions targeting Russia, Iran, and Venezuela limit or ban the ability of most U.S. companies to engage in petroleum-related transactions involving these countries.' Distinct from §VLO-rus-sup: sanctions-compliance/legal risk, not the underlying sourcing constraint itself."),

    # 2g. sector edges (energy)
    Edge("United States", ENERGY_SECTOR_NODE, 0.70, SECTOR_BROAD, "§sec-energy-us: broad US energy-regulatory/macro environment relevance. Judgment."),
    Edge("Saudi Arabia", ENERGY_SECTOR_NODE, 0.75, SECTOR_BROAD, "§sec-energy-sa: OPEC+/Saudi Arabia swing-producer decisions are widely regarded as the single dominant global-oil-price macro factor for the sector. Judgment."),
    Edge("Russia", ENERGY_SECTOR_NODE, 0.60, SECTOR_BROAD, "§sec-energy-rus: sanctions/supply-disruption sector-wide relevance, below Saudi Arabia/OPEC+'s swing-producer role. Judgment."),
    Edge(ENERGY_SECTOR_NODE, "XOM", 0.85, TRANSMIT, "§sec-xom: largest integrated major, highest sector beta of the 5. Judgment."),
    Edge(ENERGY_SECTOR_NODE, "OXY", 0.75, TRANSMIT, "§sec-oxy: large E&P, real but more geography-specific idiosyncratic risk than XOM. Judgment."),
    Edge(ENERGY_SECTOR_NODE, "EOG", 0.80, TRANSMIT, "§sec-eog: pure-play E&P, high direct oil-price beta. Judgment."),
    Edge(ENERGY_SECTOR_NODE, "LNG", 0.55, TRANSMIT, "§sec-lng: ~95% long-term fee-based contracted revenue structurally decouples LNG from spot oil/gas-price swings more than the other 4 -- most idiosyncratic of the 5, mirroring PGR's role in the financials sector. Judgment."),
    Edge(ENERGY_SECTOR_NODE, "VLO", 0.65, TRANSMIT, "§sec-vlo: refiner economics (crack spread) can move opposite to crude-price direction in some regimes, moderating sector beta relative to upstream names. Judgment."),

    # --- S3 (2026-07-23) additions: pharma_biotech sector -- see progress/sector_expansion_pharma.md ---
    # 2h. country -> stock (pharma_biotech; DEMAND/POLICY only, no manufacturing/supply
    # channel applies -- single-country US-only corridor, no citation basis researched
    # for a country-differentiated manufacturing/supply story, see note above)
    Edge("United States", "PFE", 0.59, DEMAND, "§PFE-dem: US 59.25% of FY2025 total revenue ($37,078M/$62,579M), 10-K 'Revenues by Geography' table (cited)"),
    Edge("United States", "PFE", 0.40, POLICY, "§PFE-pol: real but LOWEST of the 5 -- 10-K discloses 12 products >$1B collectively 65% of FY2025 revenue, largest single product (Eliquis) only 13% -- diversified portfolio means no single FDA action is a binary company-level event (cited, 'CONCENTRATION' note). ⚠ Judgment on magnitude."),
    Edge("United States", "VRTX", 0.63, DEMAND, "§VRTX-dem: US 62.9% of FY2025 total revenue ($7.55B/$12.0B), Q4/FY2025 earnings release geographic revenue breakdown (cited)"),
    Edge("United States", "VRTX", 0.70, POLICY, "§VRTX-pol: HIGHEST of the 5 -- CF franchise (TRIKAFTA/KAFTRIO, KALYDECO, ORKAMBI, SYMDEKO/SYMKEVI, ALYFTREK) = ~98.3% of FY2025 total revenue ($11,795.2M/$12,001.3M net of CASGEVY $115.8M + JOURNAVX $59.6M), FY2025 earnings release product-revenue table (cited) -- the single most concentrated single-franchise regulatory-binary exposure of the 5 picks."),
    Edge("United States", "BMRN", 0.37, DEMAND, "§BMRN-dem: US 37.3% of FY2025 Net Product Revenues ($1,104.97M/$2,959.25M), 10-K revenue-concentration-risk geographic table (cited) -- LOWEST of the 5, reflecting rare-disease patient populations being genuinely globally distributed rather than US-concentrated, a real structural difference from the other 4 picks' revenue-geography mix."),
    Edge("United States", "BMRN", 0.55, POLICY, "§BMRN-pol: orphan-drug/rare-disease regulatory-pathway-centric business model (10-K: multiple products' commercial exclusivity anchored in FDA/EU Orphan Drug Designation, e.g. BRINEURA/PALYNZIQ/VIMIZIM/VOXZOGO exclusivity table, cited); ROCTAVIAN's full lifecycle (FDA Complete Response Letter 2020-08-18, approved 2023-06-29, voluntary market withdrawal committed 2025-12 with ~$240.0M restructuring charge, 10-K cited) is a real, dated, disclosed regulatory-approval-decision case study specific to this ticker. ⚠ Judgment on magnitude."),
    Edge("United States", "MRNA", 0.62, DEMAND, "§MRNA-dem: US 61.7% of FY2025 total revenue ($1,199M/$1,944M), 10-K 'Total revenue by geographic area' table (cited)"),
    Edge("United States", "MRNA", 0.66, POLICY, "§MRNA-pol: dual regulatory-gate dependency distinct from the other 4 picks -- FDA approval (mRESVIA 2024; mNEXSPIKE approved 5/2025 per 10-K) AND a separate ACIP/CDC recommendation-vote gate specific to vaccines (ACIP updated RSV vaccine recommendations 2024-06-26, cited) both drive realized demand; FY2025 total revenue fell 40% YoY ($3,236M->$1,944M, 10-K cited), a real disclosed magnitude of this sector's regulatory-recommendation sensitivity. ⚠ Judgment on magnitude. Nudged twice from an initial 0.65 draft (still within the same cited/judgment-call range each time) -- 0.65 coincidentally tied JPM's own POLICY weight exactly on the dominant geopolitical_military_tension event type (byte-identical flux_score on 725/736 real days, pre-fix); the first re-draft (0.63) fixed that but coincidentally introduced a NEW tie with a 2-hop Taiwan->TSM->AMAT semiconductor path (0.8*0.90*0.35=0.252=0.4*0.63); this final value (0.66) was verified via an exhaustive all-9-event-types x all-23-tracked-tickers collision search (not just the 5 new tickers) before being accepted -- found and fixed during S3's own required pairwise-uniqueness re-check, see progress/sector_expansion_pharma.md §5."),
    Edge("United States", "REGN", 0.31, DEMAND, "§REGN-dem: EYLEA US net product sales = 31% of FY2025 total revenues (down from 42% FY2024), 10-K cited -- LOWEST direct-product-sales DEMAND weight of the 5, since a large share of REGN's revenue is collaboration-based (see §REGN-pol) rather than direct US product sales."),
    Edge("United States", "REGN", 0.61, POLICY, "§REGN-pol: ⚠ TWO-SIDED regulatory mechanism, distinct from every other pharma pick -- (a) adverse: EYLEA faces FDA-approved biosimilar competitors (Amgen's Pavblu, Sandoz's Afqlir/Eiyzey, etc., 10-K biosimilar-competitor table cited), a case where a REGULATORY approval of a COMPETITOR's product is the adverse event, not REGN's own; (b) dependency: 10-K risk factor states REGN is 'substantially dependent on our share of profits from the commercialization of Dupixent' under the Sanofi collaboration, which was 41% of FY2025 total revenues (cited, up from 32% FY2024). Judgment on magnitude, real dual mechanism. Nudged from an initial 0.60 draft (still within the same cited/judgment-call range) -- 0.60 coincidentally produced the EXACT SAME geopolitical_military_tension impact (via TWO different paths: REGN's own direct-POLICY path and country->sector->REGN path) as SCHW's DEMAND-path impact, making SCHW/REGN byte-identical on 465/736 days pre-fix -- same S1-pattern bug, found and fixed during S3's own required pairwise-uniqueness re-check (verified via the same exhaustive all-9-event-types x all-23-tickers search as MRNA's fix above), see progress/sector_expansion_pharma.md §5."),

    # 2i. sector edges (pharma_biotech)
    Edge("United States", PHARMA_SECTOR_NODE, 0.80, SECTOR_BROAD, "§sec-pharma-us: FDA is the single, near-total regulatory apparatus for this US-only-scoped sector -- even more concentrated through one country/one regulator than financials' broader bank-regulatory/macro mix. ⚠ Judgment."),
    Edge(PHARMA_SECTOR_NODE, "PFE", 0.80, TRANSMIT, "§sec-pfe: largest, most representative diversified-major name. ⚠ Judgment."),
    Edge(PHARMA_SECTOR_NODE, "VRTX", 0.55, TRANSMIT, "§sec-vrtx: single-franchise-catalyst-driven, idiosyncratic -- moves on its OWN CF/Casgevy/Journavx news more than broad sector sentiment, mirroring PGR's/LNG's role as the most decoupled pick in their own sectors. ⚠ Judgment."),
    Edge(PHARMA_SECTOR_NODE, "BMRN", 0.50, TRANSMIT, "§sec-bmrn: smaller-cap, pipeline/designation-driven rare-disease name -- most idiosyncratic of the 5, real company-specific catalysts (e.g. ROCTAVIAN's 2025 withdrawal) dominate over sector-wide moves. ⚠ Judgment."),
    Edge(PHARMA_SECTOR_NODE, "MRNA", 0.65, TRANSMIT, "§sec-mrna: highly volatile, reacts sharply to its OWN regulatory/ACIP catalysts (real 40% YoY FY2025 revenue decline, cited above) -- moderate sector-beta, not the most idiosyncratic but not the most representative either. ⚠ Judgment."),
    Edge(PHARMA_SECTOR_NODE, "REGN", 0.75, TRANSMIT, "§sec-regn: large, diversified biotech (EYLEA + Dupixent + pipeline) -- more representative of broad biotech-sector moves than the single-franchise/single-platform picks. ⚠ Judgment."),
]

# Derived index for callers that need "what edge(s) connect src->dst" directly
# (e.g. the causal-graph API endpoint's edge lookup for a stored attribution
# path) rather than running a full ExposureGraph.query() DFS. Multiple
# parallel Edge objects can share a (src, dst) pair (e.g. United States->TSM
# has manufacturing/demand/policy edges) -- all are kept, not collapsed, so
# per-channel detail survives.
EDGES_BY_PAIR: dict[tuple[str, str], list[Edge]] = {}
for _e in EDGES:
    EDGES_BY_PAIR.setdefault((_e.src, _e.dst), []).append(_e)

# --- event_type -> channel activation (docs §3; UNCALIBRATED judgment call) ---
# S3 (2026-07-23) added `regulatory_approval_decision` (pharma_biotech sector).
# ⚠ Judgment call, same discipline/caveats as the other 8 rows (uncalibrated,
# see progress/exposure_graph.md §3) -- reasoning, not just asserted:
#   POLICY=1.0 (dominant, tied with trade_export_control's own POLICY=1.0):
#     an FDA approval/CRL/advisory-committee vote/designation IS, definitionally,
#     a US federal regulatory action -- the most direct possible match to the
#     POLICY channel of any category in this table.
#   DEMAND=0.5 (elevated, matching trade_export_control's DEMAND=0.5): unlike
#     regulatory_subsidy_action (a firm-level action with only an indirect
#     revenue effect), an FDA approval/CRL directly gates whether a SPECIFIC
#     product may be sold in the US market at all -- a real, direct
#     market-access/addressable-revenue mechanism, not a secondary effect.
#   MANUFACTURING=0.1, SUPPLY=0.1 (low, matching trade_export_control's
#     MANUFACTURING=0.2/SUPPLY=0.3 but lower still): the regulatory decision
#     itself has no direct physical-production/supply-chain implication --
#     any post-approval manufacturing scale-up is a downstream
#     corporate_strategic consequence, not this event.
#   SECTOR_BROAD=0.3 (moderate-low): the decision is fundamentally
#     company/product-specific (one drug, one company), a much weaker
#     sector-wide read-through than macro/trade categories (contrast
#     macroeconomic_conditions' SECTOR_BROAD=0.9).
EVENT_CHANNEL_ACTIVATION: dict[str, dict[str, float]] = {
    "supply_chain_fab_disruption": {MANUFACTURING: 1.0, DEMAND: 0.2, POLICY: 0.1, SUPPLY: 1.0, SECTOR_BROAD: 0.3},
    "natural_disaster":            {MANUFACTURING: 1.0, DEMAND: 0.1, POLICY: 0.0, SUPPLY: 0.8, SECTOR_BROAD: 0.2},
    "geopolitical_military_tension": {MANUFACTURING: 0.8, DEMAND: 0.3, POLICY: 0.4, SUPPLY: 0.6, SECTOR_BROAD: 0.4},
    "trade_export_control":        {MANUFACTURING: 0.2, DEMAND: 0.5, POLICY: 1.0, SUPPLY: 0.3, SECTOR_BROAD: 0.5},
    "regulatory_approval_decision": {MANUFACTURING: 0.1, DEMAND: 0.5, POLICY: 1.0, SUPPLY: 0.1, SECTOR_BROAD: 0.3},
    "regulatory_subsidy_action":   {MANUFACTURING: 0.3, DEMAND: 0.2, POLICY: 0.8, SUPPLY: 0.1, SECTOR_BROAD: 0.4},
    "macroeconomic_conditions":    {MANUFACTURING: 0.1, DEMAND: 0.6, POLICY: 0.1, SUPPLY: 0.2, SECTOR_BROAD: 0.9},
    "corporate_strategic":         {MANUFACTURING: 0.3, DEMAND: 0.3, POLICY: 0.2, SUPPLY: 0.3, SECTOR_BROAD: 0.3},
    "unclassified":                {MANUFACTURING: 0.1, DEMAND: 0.1, POLICY: 0.1, SUPPLY: 0.1, SECTOR_BROAD: 0.1},
}


@dataclass
class StockExposure:
    ticker: str
    impact: float
    path: list[str]  # e.g. ["Taiwan", "TSM", "NVDA"]


class ExposureGraph:
    """Queryable weighted exposure graph. See module docstring and progress/exposure_graph.md."""

    def __init__(self, edges: list[Edge] = EDGES, nodes: dict[str, str] = NODES):
        self._nodes = dict(nodes)
        self._adj: dict[str, list[Edge]] = {}
        for e in edges:
            self._adj.setdefault(e.src, []).append(e)

    def node_type(self, name: str) -> str:
        return self._nodes[name]

    def query(
        self,
        affected_countries,
        event_type: str,
        max_hops: int = 2,
    ) -> dict[str, StockExposure]:
        """
        Given affected country/countries and an event_type, return per-stock
        propagated exposure. Impact along a path = activation(first-hop channel)
        * product(edge weights). Multiple paths to a stock are combined by MAX
        (docs §4). Graph-distance attenuation is carried by the edge weights
        themselves; no extra hop penalty (see propagation/decay.py).
        """
        if event_type not in EVENT_CHANNEL_ACTIVATION:
            raise ValueError(f"unknown event_type: {event_type!r}")
        activation = EVENT_CHANNEL_ACTIVATION[event_type]
        best: dict[str, StockExposure] = {}
        for country in affected_countries:
            if self._nodes.get(country) != COUNTRY:
                raise ValueError(f"not a country node: {country!r}")
            self._dfs(country, None, 1.0, activation, max_hops,
                      [country], {country}, best)
        return best

    def _dfs(self, node, first_channel, wprod, activation, hops_left,
             path, visited, best) -> None:
        if hops_left <= 0:
            return
        for e in self._adj.get(node, []):
            if e.dst in visited:
                continue
            fc = first_channel if first_channel is not None else e.channel
            # first hop always originates at a country node, whose edges always
            # carry a real channel, so fc is never None here.
            new_wprod = wprod * e.weight
            impact = activation[fc] * new_wprod
            new_path = path + [e.dst]
            if self._nodes[e.dst] == STOCK:
                cur = best.get(e.dst)
                if cur is None or impact > cur.impact:
                    best[e.dst] = StockExposure(e.dst, impact, new_path)
            self._dfs(e.dst, fc, new_wprod, activation, hops_left - 1,
                      new_path, visited | {e.dst}, best)


def affected_stocks(affected_countries, event_type, max_hops: int = 2):
    """Convenience: sorted (desc) list of StockExposure for an event."""
    graph = ExposureGraph()
    result = graph.query(affected_countries, event_type, max_hops=max_hops)
    return sorted(result.values(), key=lambda s: s.impact, reverse=True)
