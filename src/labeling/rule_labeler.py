"""
Keyword-dictionary rule-based labeler for text-bearing rows (RSS/NewsAPI --
anything with a non-empty title/text). See progress/event_taxonomy.md §4/§5 for
the full rubric this implements.

This is a bootstrap labeling aid, not a production classifier (per the
persona rules) -- its output quality is estimated by hand spot-check in
labeling/README.md, never presented as ground truth.

All dictionaries below are module-level, plain Python dicts/lists --
deliberately inspectable/editable, not compiled into opaque logic, so a
reviewer or future editor can read the actual keyword lists directly.
"""

from __future__ import annotations

import functools
import re
from typing import Optional

from config.mvp_scope import ALL_COMPANY_NAMES, TRACKED_STOCKS, sector_for_ticker
from src.labeling.schema import ClassifiedEvent, severity_band_midpoint

LABELER_VERSION = "rule_labeler_v1"

# --- Event-type keyword dictionary -----------------------------------------
# Order here matches labeling.schema.EVENT_TYPES, which is itself the
# documented tie-break order for co-matching categories (see
# progress/event_taxonomy.md ambiguity 2: trade_export_control is checked/wins
# before geopolitical_military_tension / regulatory_subsidy_action on ties).
# S3 (2026-07-23) inserted `regulatory_approval_decision` BEFORE
# `regulatory_subsidy_action` for the same reason -- see progress/event_taxonomy.md
# "Flagged ambiguity 7" and labeling/schema.py's EVENT_TYPES comment.
EVENT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "trade_export_control": [
        "export control", "export controls", "export ban", "export restriction",
        "entity list", "tariff", "tariffs", "sanction", "sanctions",
        "trade restriction", "trade war", "licensing regime", "export license",
        "chip ban", "commerce department", "bis rule",
    ],
    # Bare "military" removed (2026-08-16) -- see labeling/README.md's
    # rule_labeler.py bug writeup. gdelt_content_filter.py's
    # _TOO_GENERIC_FOR_HEADLINES set already excluded "military"/"north korea"
    # for the GDELT headline-only path, on the assumption that full-article
    # text here dilutes a generic hit enough to be safe. Live evidence
    # disproves that for "military": of 26 rule_based geopolitical_military_
    # tension rows, 5 matched "military", and in the 3 where it was the SOLE
    # matched keyword, all 3 (100%) were false positives -- a BTS/K-pop
    # article about mandatory South Korean conscription, an unrelated HELOC
    # personal-finance article quoting "military service in Afghanistan" in a
    # burn-pit aside, and a WWII historical retrospective. Worse than the
    # GDELT precedent (22.6% FP) that justified exclusion there. The 2 rows
    # where "military" co-occurred with "drills" (genuine Pyongyang/Cambodia
    # military-drills stories) are unaffected by this removal since "drills"
    # alone still wins the category.
    #
    # "north korea" is deliberately left untouched here -- it has ZERO live
    # matches in this label source to quantify against (unlike "military"),
    # so excluding it would violate this project's quantify-before-excluding
    # discipline (see bugs 3 and 7 in labeling/README.md). Re-quantify if a
    # live rule_based counter-example for "north korea" ever surfaces.
    "geopolitical_military_tension": [
        "invasion", "invade", "war", "missile", "warship",
        "cross-strait", "taiwan strait", "north korea", "provocation",
        "troop", "troops", "conflict", "blockade", "drill", "drills",
        "naval", "airspace incursion", "tension", "tensions",
    ],
    "supply_chain_fab_disruption": [
        "fab outage", "power outage", "water shortage", "chip shortage",
        "factory fire", "cyberattack", "cyber attack", "production halt",
        "supply chain disruption", "logistics disruption", "plant closure",
        "shuts down production", "halts production", "fab shutdown",
    ],
    "natural_disaster": [
        "earthquake", "typhoon", "flood", "flooding", "drought", "tsunami",
        "landslide", "wildfire", "seismic",
    ],
    # S3 (2026-07-23) addition -- see progress/sector_expansion_pharma.md and
    # progress/event_taxonomy.md "Flagged ambiguity 7". Keywords deliberately name
    # the SPECIFIC drug/biologic regulatory decision or pathway (approval, CRL,
    # trial readout, designation, advisory-committee vote), not generic
    # "regulator"/"regulatory" language -- that stays in regulatory_subsidy_action
    # below, per the boundary this category draws (product-specific regulatory
    # decision vs. firm-level regulatory/subsidy action). Refined against this
    # project's own research pass on FDA-approval news framing (see
    # progress/sector_expansion_pharma.md's ticker research), not assumed.
    "regulatory_approval_decision": [
        "fda approval", "fda approves", "fda approved", "fda rejects",
        "fda rejection", "complete response letter", "advisory committee",
        "adcomm", "breakthrough therapy designation", "priority review",
        "orphan drug designation", "fast track designation",
        "biologics license application", "new drug application",
        "phase 3 trial", "phase 2 trial", "clinical trial results",
        "topline results", "trial readout", "accelerated approval",
        "label expansion", "regenerative medicine advanced therapy",
    ],
    "regulatory_subsidy_action": [
        "chips act", "subsidy", "subsidies", "grant", "antitrust",
        "regulatory ruling", "regulator", "fine", "fined", "compliance order",
        "investigation opened",
    ],
    "macroeconomic_conditions": [
        "interest rate", "interest rates", "inflation", "gdp", "federal reserve",
        "the fed", "rate hike", "rate cut", "recession", "demand outlook",
        "inventory cycle", "economic growth",
    ],
    "corporate_strategic": [
        "acquisition", "acquire", "acquires", "merger", "earnings", "guidance",
        "ceo", "executive", "product launch", "capacity expansion",
        "quarterly results", "ipo", "stock buyback", "layoffs",
    ],
}

# --- Severity keyword bands -------------------------------------------------
# Checked from band 5 down to band 1; first hit wins. If nothing matches but
# an event_type was found, default is band 2 (flagged weak default, see
# progress/event_taxonomy.md §4).
SEVERITY_KEYWORDS: dict[int, list[str]] = {
    5: [
        "war", "invasion", "invades", "military conflict", "full embargo",
        "total embargo", "complete embargo", "declares war",
    ],
    4: [
        "export ban", "banned", "enacted", "acquisition", "to acquire",
        "acquires", "merger", "earthquake", "trade war", "escalates",
        "escalation", "halts production", "shuts down", "shutdown",
    ],
    3: [
        "proposed", "proposal", "investigation", "probe", "temporarily closed",
        "temporary closure", "guidance miss", "delay", "delayed", "tariff",
    ],
    2: [
        "minor", "small", "incremental", "slight",
    ],
    1: [
        "recap", "roundup", "round-up", "weekly review", "explainer",
        "quarterly review", "analyst commentary",
    ],
}
_DEFAULT_SEVERITY_BAND = 2  # weak default, see progress/event_taxonomy.md §4

# --- Country aliasing --------------------------------------------------------
# Aliases documented inline, editable. The bare "US" alias is intentionally
# matched as a whole *uppercase* token only (\bUS\b, case-sensitive) to avoid
# false positives on the common lowercase pronoun "us" -- see
# progress/event_taxonomy.md §5.
_COUNTRY_ALIASES: dict[str, list[str]] = {
    "United States": ["united states", "u.s.", "america", "washington"],
    "Taiwan": ["taiwan", "taipei"],
    "South Korea": ["south korea", "korea", "seoul"],
    # S2 (2026-07-23) additions -- see progress/sector_expansion_energy.md. "saudi" alone is kept
    # (not just "saudi arabia") since "Saudi" is used as a standalone adjective in real energy
    # reporting ("Saudi oil minister", "Saudi crude") with negligible false-positive risk (no
    # common unrelated English usage, unlike the "Progressive"/"Schwab" collision risks already
    # documented for the financials sector). "russia"/"russian"/"moscow"/"kremlin" cover the
    # same real-world phrasing variety GDELT-adjacent RSS/NewsAPI energy reporting would use.
    "Saudi Arabia": ["saudi arabia", "saudi", "riyadh"],
    "Russia": ["russia", "russian", "moscow", "kremlin"],
}
_US_TOKEN_RE = re.compile(r"\bUS\b")

# --- Sector keywords ---------------------------------------------------------
# S0 REWORK: generalized from a single flat keyword list to one list PER
# registered sector (see config/mvp_scope.py's SECTORS registry). Only
# "semiconductors" has entries until S1/S2/S3 (financials/energy/pharma_biotech)
# add their own -- adding a new sector here is a pure data addition, no
# `_match_sector` logic change needed.
SECTOR_KEYWORDS: dict[str, list[str]] = {
    "semiconductors": [
        "semiconductor", "semiconductors", "chipmaker", "chipmakers",
        "foundry", "foundries", "wafer", "wafers", "chip",
    ],
    # S1 (2026-07-23) addition -- see progress/sector_expansion_financials.md. Generic
    # financial-sector terms, same word-boundary-matched-keyword style as above; these
    # are only consulted as a fallback when no tracked company name matched (see
    # _match_sector's docstring), so a bare "bank"/"lender" hit never overrides an
    # unambiguous ticker/company-name match.
    "financials": [
        "bank", "banks", "banking", "lender", "lenders", "lending",
        "insurer", "insurers", "insurance", "underwriting", "underwriter",
        "financial services", "regional bank", "regional banks",
        "credit card", "brokerage",
    ],
    # S2 (2026-07-23) addition -- see progress/sector_expansion_energy.md. Structural/business-model
    # energy terms, same word-boundary-matched-keyword style as above; fallback-only, same
    # discipline as the financials list (never overrides an unambiguous ticker/company match).
    # Deliberately excludes bare "oil"/"gas" (too generic/high false-positive risk across
    # unrelated financial news -- "gas" alone would collide with "natural gas" used as a filler
    # phrase, cost commentary, etc.) in favor of more specific multi-word/structural terms.
    "energy": [
        "crude oil", "opec", "opec+", "refinery", "refineries", "refining",
        "pipeline", "pipelines", "upstream", "midstream", "downstream",
        "lng", "liquefied natural gas", "barrel", "barrels", "oilfield",
        "petroleum", "shale", "wellhead", "crack spread",
    ],
    # S3 (2026-07-23) addition -- see progress/sector_expansion_pharma.md. Structural/
    # business-model pharma/biotech terms, same word-boundary-matched-keyword style
    # as above; fallback-only, same discipline as the other 3 sectors' lists (never
    # overrides an unambiguous ticker/company-name match). Deliberately excludes
    # bare "drug"/"trial" (too generic/high false-positive risk -- "drug" collides
    # with unrelated crime/legal-news usage, "trial" with unrelated legal-proceeding
    # usage) in favor of more specific multi-word/structural terms, same discipline
    # as energy's exclusion of bare "oil"/"gas".
    "pharma_biotech": [
        "biopharmaceutical", "biotech", "biotechnology", "pharmaceutical",
        "pharmaceuticals", "drugmaker", "drugmakers", "clinical trial",
        "clinical-stage", "cystic fibrosis", "rare disease", "gene therapy",
        "biosimilar", "vaccine", "vaccines", "oncology drug",
    ],
}

_TICKER_NAME_PAIRS = [
    (stock["ticker"], name.lower()) for stock in TRACKED_STOCKS for name in stock["company_names"]
]


@functools.lru_cache(maxsize=None)
def _keyword_regex(keyword: str) -> re.Pattern:
    # Word-boundary match, not bare substring -- a naive `keyword in haystack`
    # check produces real false positives found in the Phase B spot-check,
    # e.g. "war" inside "Toward", or the ticker alias "AMD" inside "Mamdani".
    # See labeling/README.md for the concrete examples this fixed.
    #
    # Deliberately uses (?<!\w)...(?!\w) lookarounds rather than \b\b: a plain
    # \b fails on aliases that end in punctuation (e.g. "u.s.") because \b
    # requires an actual word/non-word *transition*, and "." followed by a
    # space is non-word on both sides -- not a transition. The lookaround
    # form only asserts "not immediately adjacent to a word character" on
    # each side, which is what we actually want here.
    return re.compile(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)", re.IGNORECASE)


def _contains_keyword(haystack: str, keyword: str) -> bool:
    return _keyword_regex(keyword).search(haystack) is not None


def _match_event_type(haystack: str) -> tuple[str, int, list[str]]:
    """
    Returns (event_type, hit_count, matched_keywords) for the best-scoring
    category, walking EVENT_TYPES order for deterministic tie-breaks.
    "unclassified" with hit_count 0 if nothing matched.
    """
    best_type = "unclassified"
    best_count = 0
    best_keywords: list[str] = []
    for event_type, keywords in EVENT_TYPE_KEYWORDS.items():
        matched = [kw for kw in keywords if _contains_keyword(haystack, kw)]
        if len(matched) > best_count:
            best_type = event_type
            best_count = len(matched)
            best_keywords = matched
    return best_type, best_count, best_keywords


def _match_severity(haystack: str) -> tuple[Optional[int], Optional[str]]:
    for band in (5, 4, 3, 2, 1):
        for kw in SEVERITY_KEYWORDS[band]:
            if _contains_keyword(haystack, kw):
                return band, kw
    return None, None


def _match_countries(original_text: str, lower_haystack: str) -> list[str]:
    found = set()
    for country, aliases in _COUNTRY_ALIASES.items():
        if any(_contains_keyword(lower_haystack, alias) for alias in aliases):
            found.add(country)
    if _US_TOKEN_RE.search(original_text):
        found.add("United States")
    return sorted(found)


def _match_sector(haystack: str) -> Optional[str]:
    """
    S0 REWORK: a matched company/ticker name is the strongest, unambiguous
    signal -- resolve its sector directly via sector_for_ticker() rather than
    returning a single hardcoded sector regardless of which company matched
    (the old single-sector behavior, which would have silently mislabeled
    e.g. a financials-sector article as "semiconductors" once a second sector
    existed). Only falls back to the generic per-sector keyword lists
    (SECTOR_KEYWORDS) when no tracked company is named at all. If more than
    one sector's generic keywords match (not possible today with one sector
    registered), the first match in SECTORS registration order wins --
    deterministic, but a real ambiguity worth flagging once genuinely
    cross-sector generic terms exist.
    """
    for ticker, name in _TICKER_NAME_PAIRS:
        if _contains_keyword(haystack, name):
            return sector_for_ticker(ticker)
    for sector_name, keywords in SECTOR_KEYWORDS.items():
        if any(_contains_keyword(haystack, kw) for kw in keywords):
            return sector_name
    return None


def _match_companies(haystack: str) -> list[str]:
    tickers = {ticker for ticker, name in _TICKER_NAME_PAIRS if _contains_keyword(haystack, name)}
    return sorted(tickers)


def label(raw_event: dict) -> ClassifiedEvent:
    """
    Pure function: raw_events row (as a dict) -> ClassifiedEvent. Handles
    rows with empty title+text by emitting unclassified/confidence 0 rather
    than crashing or guessing.
    """
    title = raw_event.get("title") or ""
    text = raw_event.get("text") or ""
    combined = f"{title} {text}".strip()

    if not combined:
        return ClassifiedEvent(
            event_id=raw_event["id"],
            label_source="rule_based",
            event_type="unclassified",
            severity=1,
            severity_score=severity_band_midpoint(1),
            confidence=0.0,
            polarity=None,
            countries=[],
            sector=None,
            companies=[],
            reasoning="no title/text available for keyword matching",
            labeler_version=LABELER_VERSION,
        )

    lower = combined.lower()

    event_type, hit_count, matched_keywords = _match_event_type(lower)

    if event_type == "unclassified":
        severity = 1
        confidence = 0.0
        reasoning = "no event-type keyword matched title/text"
    else:
        severity_hit, severity_kw = _match_severity(lower)
        if severity_hit is None:
            severity = _DEFAULT_SEVERITY_BAND
            severity_note = (
                "no severity keyword matched; defaulted to band 2 "
                "(weak default, see progress/event_taxonomy.md §4)"
            )
        else:
            severity = severity_hit
            severity_note = f"severity keyword matched: {severity_kw!r}"

        confidence = round(min(0.25 + 0.10 * hit_count, 0.85), 4)
        reasoning = (
            f"event_type keyword(s) matched: {matched_keywords!r} "
            f"({hit_count} hit(s)); {severity_note}"
        )

    return ClassifiedEvent(
        event_id=raw_event["id"],
        label_source="rule_based",
        event_type=event_type,
        severity=severity,
        severity_score=severity_band_midpoint(severity),
        confidence=confidence,
        polarity=None,  # not attempted by the rule-labeler, see progress/event_taxonomy.md §4
        countries=_match_countries(combined, lower),
        sector=_match_sector(lower),
        companies=_match_companies(lower),
        reasoning=reasoning,
        labeler_version=LABELER_VERSION,
    )


def label_batch(raw_events: list[dict]) -> list[ClassifiedEvent]:
    return [label(e) for e in raw_events]
