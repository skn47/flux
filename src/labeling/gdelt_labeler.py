"""
Deterministic GDELT-derived labeler. See progress/event_taxonomy.md §3 for the
full design rationale and citations behind every transform below -- this
module is a direct, mechanical implementation of that spec, not a place to
make new judgment calls.

Reads the `raw_metadata` dict already stored by `ingestion/gdelt.py` for each
`raw_events` row (GoldsteinScale, QuadClass, NumMentions, actor/action
country codes). Does NOT re-fetch anything from GDELT.

Hard limitation (progress/event_taxonomy.md ambiguity 5): GDELT's CAMEO codes
describe diplomatic/military actor interactions, not economic/sectoral
content. This labeler can therefore only ever emit `geopolitical_military_tension`
or `unclassified` as event_type -- never any of the other 5 categories --
and always emits sector=None, companies=[] with a reasoning string
explaining why, since GDELT rows carry no article text.
"""

from __future__ import annotations

import math
from typing import Optional

from config.mvp_scope import COUNTRIES
from src.labeling.schema import ClassifiedEvent, severity_band

LABELER_VERSION = "gdelt_labeler_v1"

# NumMentions calibration cap: log1p(20) denominator, per progress/event_taxonomy.md
# §3, calibrated against the live corpus's observed median (4) / max (34)
# NumMentions at design time -- an explicitly flagged judgment call (ambiguity 6),
# not an empirically-fit constant.
_MENTIONS_CAP = 20
_GOLDSTEIN_WEIGHT = 0.65
_MENTIONS_WEIGHT = 0.35

_SECTOR_COMPANY_REASON = (
    "sector/companies not derivable from GDELT actor-interaction data (no "
    "article text; the GDELT connector filters only by actor country, not "
    "by sector) -- always null/[] for gdelt_derived labels, per "
    "progress/event_taxonomy.md §3."
)

# CAMEO code (3-letter, Actor1/2CountryCode) and FIPS code (2-letter,
# ActionGeo_CountryCode) both map to the same canonical country name, reusing
# config.mvp_scope.COUNTRIES rather than re-deriving code tables by hand.
_CAMEO_TO_NAME = {c["cameo"]: c["name"] for c in COUNTRIES}
_FIPS_TO_NAME = {c["fips10_4"]: c["name"] for c in COUNTRIES}


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _resolve_countries(raw_metadata: dict) -> list[str]:
    codes = [
        raw_metadata.get("Actor1CountryCode"),
        raw_metadata.get("Actor2CountryCode"),
    ]
    names = {_CAMEO_TO_NAME[c] for c in codes if c in _CAMEO_TO_NAME}

    action_geo = raw_metadata.get("ActionGeo_CountryCode")
    if action_geo in _FIPS_TO_NAME:
        names.add(_FIPS_TO_NAME[action_geo])

    return sorted(names)


def label(raw_event: dict) -> ClassifiedEvent:
    """
    Pure function: raw_events row (as a dict, with raw_metadata already
    parsed -- see labeling.storage.LabelStore.unlabeled_raw_events) ->
    ClassifiedEvent. Never raises on missing/malformed metadata fields;
    treats them as "not available" rather than crashing the batch.
    """
    meta = raw_event.get("raw_metadata") or {}

    quad_class = meta.get("QuadClass")
    goldstein: Optional[float] = meta.get("GoldsteinScale")
    num_mentions = meta.get("NumMentions") or 0

    if quad_class in (3, 4):
        event_type = "geopolitical_military_tension"
        confidence = 0.55 if quad_class == 4 else 0.40
        type_reason = (
            f"GDELT QuadClass={quad_class} "
            f"({'Material' if quad_class == 4 else 'Verbal'} Conflict) -- "
            "conflictual actor interaction between MVP-country actors, per "
            "progress/event_taxonomy.md §3 type transform."
        )
    else:
        event_type = "unclassified"
        confidence = 0.15
        type_reason = (
            f"GDELT QuadClass={quad_class} (cooperative/verbal-statement class); "
            "actor-interaction codes alone don't reliably indicate a "
            "sector-relevant sub-category (trade/regulatory/corporate) "
            "without article text, per progress/event_taxonomy.md §3."
        )

    if goldstein is not None:
        polarity = _clip(goldstein / 10.0, -1.0, 1.0)
        goldstein_component = _clip(-goldstein / 10.0, 0.0, 1.0)
    else:
        # Defensive fallback only -- GDELT rows should always carry
        # GoldsteinScale in practice. Treated as "no conflict signal
        # available" (component 0) rather than fabricating a value.
        polarity = None
        goldstein_component = 0.0

    mentions_component = _clip(
        math.log1p(num_mentions) / math.log1p(_MENTIONS_CAP), 0.0, 1.0
    )
    severity_score = round(
        _clip(
            _GOLDSTEIN_WEIGHT * goldstein_component + _MENTIONS_WEIGHT * mentions_component,
            0.0,
            1.0,
        ),
        4,
    )
    severity = severity_band(severity_score)

    countries = _resolve_countries(meta)

    reasoning = f"{type_reason} {_SECTOR_COMPANY_REASON}"

    return ClassifiedEvent(
        event_id=raw_event["id"],
        label_source="gdelt_derived",
        event_type=event_type,
        severity=severity,
        severity_score=severity_score,
        confidence=confidence,
        polarity=polarity,
        countries=countries,
        sector=None,
        companies=[],
        reasoning=reasoning,
        labeler_version=LABELER_VERSION,
    )


def label_batch(raw_events: list[dict]) -> list[ClassifiedEvent]:
    return [label(e) for e in raw_events]
