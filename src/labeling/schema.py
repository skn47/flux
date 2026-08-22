"""
Shared classification schema for Phase B labeling.

Every labeler (GDELT-derived, rule-based, LLM-assisted, ...) normalizes its
output into a single `ClassifiedEvent`. This module is the single source of
truth for the closed taxonomy and severity bands documented in
`progress/event_taxonomy.md` -- labelers import these constants instead of
re-typing string literals, so a taxonomy edit only has to happen in one place.

See `progress/event_taxonomy.md` for the full design rationale, citations, and
flagged ambiguities behind every choice below. This file intentionally
contains no judgment calls of its own -- it's a mechanical restatement of
that document's §1/§2/§5 as importable Python constants + a plain dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

# --- Event types (closed 8 + unclassified fallback) -----------------------
# Order matters: labelers that need a deterministic tie-break between
# co-matching categories (see progress/event_taxonomy.md ambiguity 2) walk this
# list in order, so trade_export_control is checked before the categories it
# is documented to take priority over.
#
# S3 (2026-07-23, pharma_biotech sector addition) added
# `regulatory_approval_decision` -- placed BEFORE `regulatory_subsidy_action`,
# matching the existing precedent set by `trade_export_control`'s position
# ahead of `geopolitical_military_tension`/`regulatory_subsidy_action`: the
# more PRODUCT/DECISION-specific category is checked (and wins keyword-count
# ties) before the broader, firm-level regulatory category. See
# progress/event_taxonomy.md's new "Flagged ambiguity 7" for the full boundary
# reasoning (regulatory_approval_decision = a regulator's own decision about
# whether/how a SPECIFIC drug/biologic may be marketed; regulatory_subsidy_action
# = broader firm-level subsidy/antitrust/non-trade regulatory actions).
EVENT_TYPES: list[str] = [
    "trade_export_control",
    "geopolitical_military_tension",
    "supply_chain_fab_disruption",
    "natural_disaster",
    "regulatory_approval_decision",
    "regulatory_subsidy_action",
    "macroeconomic_conditions",
    "corporate_strategic",
    "unclassified",
]

# --- Label provenance -------------------------------------------------------
LABEL_SOURCES: list[str] = [
    "gdelt_derived", "rule_based", "llm_assisted", "manual",
    # Added for the GDELT geopolitical_military_tension content-check pass
    # (progress/event_taxonomy.md) -- labeling/ollama_labeler.py's local-model
    # stage-2 corrections. See flux_engine/formula.py RELIABILITY for its
    # weight.
    "ollama_assisted",
]

# --- Severity anchor bands (severity_score -> severity int + name) --------
# (severity int, score_low_inclusive, score_high, name). Last band's high end
# is inclusive of 1.0; all others are exclusive of their upper bound.
SEVERITY_BANDS: list[tuple[int, float, float, str]] = [
    (1, 0.00, 0.15, "negligible"),
    (2, 0.15, 0.35, "minor"),
    (3, 0.35, 0.55, "moderate"),
    (4, 0.55, 0.75, "major"),
    (5, 0.75, 1.01, "severe"),  # 1.01 upper bound makes score==1.0 land in band 5
]

SEVERITY_NAMES: dict[int, str] = {band: name for band, _, _, name in SEVERITY_BANDS}


def severity_band(score: float) -> int:
    """Map a continuous severity_score in [0, 1] to its anchor band (1-5)."""
    clipped = max(0.0, min(1.0, score))
    for band, low, high, _name in SEVERITY_BANDS:
        if low <= clipped < high:
            return band
    return SEVERITY_BANDS[-1][0]  # defensive fallback for score == 1.0 edge case


def severity_band_midpoint(band: int) -> float:
    """
    Inverse-ish helper: representative severity_score for a band chosen
    directly (e.g. by the rule-labeler's discrete keyword matching, which
    picks a band first rather than computing a continuous score). Returns the
    midpoint of the band's score range.
    """
    for b, low, high, _name in SEVERITY_BANDS:
        if b == band:
            capped_high = min(high, 1.0)
            return round((low + capped_high) / 2, 4)
    raise ValueError(f"Unknown severity band: {band!r}")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ClassifiedEvent:
    event_id: str                        # FK -> raw_events.id
    label_source: str                    # one of LABEL_SOURCES
    event_type: str                      # one of EVENT_TYPES
    severity: int                        # 1-5, see SEVERITY_BANDS
    severity_score: float                # 0.0-1.0 continuous
    confidence: float                    # 0.0-1.0
    polarity: Optional[float] = None     # -1.0..+1.0, None when not derivable
    countries: list[str] = field(default_factory=list)   # canonical country names
    sector: Optional[str] = None
    companies: list[str] = field(default_factory=list)   # tracked tickers
    reasoning: Optional[str] = None
    labeler_version: Optional[str] = None
    labeled_at: str = field(default_factory=now_iso)

    def __post_init__(self) -> None:
        if self.label_source not in LABEL_SOURCES:
            raise ValueError(f"Unknown label_source: {self.label_source!r}")
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown event_type: {self.event_type!r}")
        if self.severity not in SEVERITY_NAMES:
            raise ValueError(f"severity must be 1-5, got {self.severity!r}")

    def as_dict(self) -> dict:
        return asdict(self)
