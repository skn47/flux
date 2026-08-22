"""
Stage 2 (local LLM) classifier for `gdelt_derived` events that Stage 1
(labeling/gdelt_content_filter.py) couldn't confidently confirm or reject.

Mirrors labeling/llm_labeler.py's shape (same "confidence must honestly reflect
uncertainty" instruction, same per-row try/except so one bad row never aborts
a batch) but targets a local Ollama server over plain HTTP (via `requests`,
already a project dependency) instead of the Anthropic API -- no cost, no
ANTHROPIC_API_KEY needed, per the user's explicit direction to use a free
local model for this pass.

Unlike llm_labeler.py, this module does NOT expose the full closed taxonomy to
the model -- event_type is constrained to a binary
{geopolitical_military_tension, unclassified} (see `_EVENT_TYPES_STAGE2`
below). Two earlier attempts at letting this 3B model pick among the full
taxonomy both produced unreliable results (see that constant's comment and
labeling/README.md) -- the binary judgment is the one thing it has been
consistently accurate at.

Ollama itself is not installed by this module -- see progress/event_taxonomy.md /
labeling/README.md for the install command. This module only talks to
whatever server is already running at OLLAMA_HOST; if it isn't reachable or
the model isn't pulled, `label_batch` returns [] with a logged warning and
makes no further calls, same graceful-degradation discipline as
llm_labeler.py's ANTHROPIC_API_KEY gate.

GDELT rows have no article body (see ingestion/gdelt.py) -- only the headline
scraped by ingestion/article_titles.py is available, so `raw_event["text"]`
is always empty here; the model classifies on title alone.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import requests

from config.mvp_scope import COUNTRIES, SECTORS, TRACKED_TICKERS
from src.labeling.schema import ClassifiedEvent, now_iso

logger = logging.getLogger("labeling.ollama_labeler")

LABELER_VERSION = "ollama_labeler_v1"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b-instruct"
REQUEST_TIMEOUT = 60
REACHABILITY_TIMEOUT = 3

_COUNTRY_NAMES = [c["name"] for c in COUNTRIES]
_SECTOR_NAMES = list(SECTORS.keys())

# Deliberately NOT the full closed taxonomy. Two earlier attempts (bare category
# names, then names + one-line definitions) both let this 3B model pick a
# specific alternate category -- and it did so unreliably: 108/370 headlines in
# one rerun got "natural_disaster" including plain car crashes, a drunk-driving
# crash, a bridge collapse, and a shooting, with the model's own `reasoning`
# field sometimes directly contradicting its own event_type pick (e.g. "a
# human-caused incident... consistent with a natural disaster category"). A
# small local model can state a negative constraint correctly in prose and
# still violate it in the structured field right next to it -- so the fix here
# is to remove the failure mode by construction, not to word the prompt more
# carefully a third time. The binary judgment below (still-geopolitical vs.
# not) is the one thing this model has been reliably accurate at throughout
# this module's spot-checks (see labeling/README.md). Any real re-categorization
# into supply_chain_fab_disruption/natural_disaster/etc. is left to a future,
# separately-validated pass -- not attempted here.
_EVENT_TYPES_STAGE2 = ["geopolitical_military_tension", "unclassified"]

# 2026-08-16: Stage 1's "confirm" verdict used to be trusted outright with no
# numeric score at all (labeling/run_gdelt_reclassify.py's old fast path) --
# the direct root cause of at least two false-positive bugs found this session
# (bare "troop"/"troops" and "military" keyword hits with zero content
# validation). Stage 1 confirms now route through here like "uncertain" rows
# do, and MUST clear this confidence floor to count as a real confirm; below
# it, they're treated as a reject regardless of event_type. Threshold chosen
# from the live confirm-confidence distribution at the time (598 historical
# ollama_assisted confirms): 0.7->7 rows, 0.8->352, 0.9->238, 1.0->1 -- 0.75
# cuts only the thin bottom tail, leaving the bulk of already-spot-checked-good
# Stage 2 output untouched. A starting point, not a definitive number -- retune
# against labeling/README.md's spot-check methodology if the false-positive/
# false-negative balance drifts.
MIN_CONFIRM_CONFIDENCE = 0.75

# An earlier version of the prompt below also included "When in doubt, choose
# unclassified" (added while the model still had the full taxonomy available,
# to counter it guessing a wrong specific category). Once event_type became
# binary, that phrase started producing false negatives on unambiguous
# military content instead -- e.g. "North Korea fires salvo of short-range
# ballistic missiles" and "North Korea tests rocket launcher in threat to
# Seoul" both rejected, with the model's own reasoning self-contradicting
# ("about missile launches... but does not specify... conflict risk"). A/B
# tested removing the phrase against both the new false negatives and the
# original false-positive titles (car crash, oil price, tourism reopening) --
# it recovered some of the former with no regression on the latter, so it's
# gone. KNOWN REMAINING LIMITATION even after removing it (see
# labeling/README.md): the model still misses some unambiguous military
# headlines that don't explicitly name a tracked corridor/adversary
# relationship (a missile test "threatening Seoul" without the sentence
# spelling out why that's a security risk). This looks like a real
# capability ceiling of a 3B local model rather than a promptable bug --
# false NEGATIVES are the safe failure direction here (a missed real event
# just doesn't compete for the cluster, vs. a false positive injecting wrong
# signal), so it's disclosed as a limitation rather than chased further.

_SYSTEM_PROMPT = f"""You are a careful financial-event classification assistant for a \
multi-sector flux-monitoring pipeline (tracked sectors: {_SECTOR_NAMES}). You are given a \
NEWS HEADLINE ONLY (no article body) that GDELT's automated event coder flagged as \
"geopolitical_military_tension" purely from actor/action codes, without reading the headline. \
Your job is a real semantic check: does this headline actually describe military/security \
posturing or conflict RISK to a tracked corridor (cross-strait, Korea peninsula, US-China/Russia/\
Saudi Arabia friction)? Or is it a false positive -- domestic crime, an accident, entertainment/\
consumer content, a natural disaster, an economic/market story, or a routine civil/legal matter \
that superficially coded as "conflict" because CAMEO's action codes don't distinguish state \
violence from ordinary crime?

This is a BINARY decision -- event_type must be exactly one of:
- "geopolitical_military_tension": the headline itself, read plainly, is about military/security \
posturing or conflict risk to a tracked corridor.
- "unclassified": anything else -- including headlines that are clearly about some OTHER real \
topic (a natural disaster, an accident, a market/earnings story, a regulatory action). Do not try \
to guess which other category fits; "unclassified" is correct for all of them here.

Only report countries from this fixed set if genuinely relevant: {_COUNTRY_NAMES}. Only report sector as one of \
{_SECTOR_NAMES} if genuinely about that industry, else null. Only report companies as tickers from \
this fixed set if genuinely implicated: {TRACKED_TICKERS}. Never fabricate an entity not supported \
by the headline.

Severity is an anchored 1-5 band (1=negligible/routine, 5=severe/existential e.g. war, invasion, \
total embargo); severity_score is a continuous 0.0-1.0 estimate consistent with that band. polarity \
is -1.0 (very adverse to the corridor) to +1.0 (favorable), or null if you cannot tell direction. \
confidence (0.0-1.0) must honestly reflect your own uncertainty from a headline alone -- do not \
default to a high number. reasoning must be ONE short sentence (under 20 words)."""

_RESPONSE_FORMAT = {
    "type": "object",
    "properties": {
        "event_type": {"type": "string", "enum": _EVENT_TYPES_STAGE2},
        "severity": {"type": "integer", "minimum": 1, "maximum": 5},
        "severity_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "polarity": {"type": ["number", "null"], "minimum": -1.0, "maximum": 1.0},
        "countries": {"type": "array", "items": {"type": "string"}},
        "sector": {"type": ["string", "null"]},
        "companies": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string"},
    },
    "required": [
        "event_type", "severity", "severity_score", "confidence", "reasoning",
    ],
}


def is_available(host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL) -> bool:
    """Reachability + model-pulled check. Never raises."""
    try:
        resp = requests.get(f"{host}/api/tags", timeout=REACHABILITY_TIMEOUT)
        resp.raise_for_status()
        names = {m.get("name") for m in resp.json().get("models", [])}
        return model in names
    except requests.RequestException:
        return False


def _clip(value: Optional[float], lo: float, hi: float, default: float) -> float:
    """Clip into [lo, hi], with a defensive rescale for the 0-1-vs-0-100
    confusion small local models are prone to (observed during this module's
    design): a value > hi is assumed to be on a 0-100 scale and divided down
    before clipping, rather than just clamped to hi (which would silently
    turn a garbled low-confidence read into a false "fully confident")."""
    if value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v > hi and hi <= 1.0:
        v = v / 100.0
    return max(lo, min(hi, v))


def _label_one(
    raw_event: dict, host: str, model: str
) -> Optional[ClassifiedEvent]:
    title = (raw_event.get("title") or "").strip()
    if not title:
        logger.warning("ollama_labeler: skipping event_id=%s, no title", raw_event.get("id"))
        return None

    try:
        resp = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": title},
                ],
                "format": _RESPONSE_FORMAT,
                "stream": False,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        logger.warning(
            "ollama_labeler: request failed for event_id=%s: %s", raw_event.get("id"), exc
        )
        return None

    try:
        parsed = json.loads(content)
    except ValueError:
        logger.warning(
            "ollama_labeler: model returned non-JSON for event_id=%s: %r",
            raw_event.get("id"), content[:200],
        )
        return None

    event_type = parsed.get("event_type")
    if event_type not in _EVENT_TYPES_STAGE2:
        logger.warning(
            "ollama_labeler: model returned unknown event_type=%r for event_id=%s",
            event_type, raw_event.get("id"),
        )
        return None

    # Small local models are inconsistent about the 0-1 vs 0-100 confidence
    # scale (observed during this module's design) -- clip defensively rather
    # than trust the raw value, same spirit as llm_labeler.py's pydantic
    # Field(ge=..., le=...) validation but tolerant instead of rejecting.
    severity_score = _clip(parsed.get("severity_score"), 0.0, 1.0, 0.5)
    confidence = _clip(parsed.get("confidence"), 0.0, 1.0, 0.5)
    severity = parsed.get("severity")
    if not isinstance(severity, int) or not (1 <= severity <= 5):
        severity = max(1, min(5, round(severity_score * 4) + 1))

    return ClassifiedEvent(
        event_id=raw_event["id"],
        label_source="ollama_assisted",
        event_type=event_type,
        severity=severity,
        severity_score=severity_score,
        confidence=confidence,
        polarity=parsed.get("polarity"),
        countries=[c for c in parsed.get("countries", []) if isinstance(c, str)],
        sector=parsed.get("sector"),
        companies=[c for c in parsed.get("companies", []) if isinstance(c, str)],
        reasoning=parsed.get("reasoning"),
        labeler_version=f"{LABELER_VERSION}:{model}",
        labeled_at=now_iso(),
    )


def label_batch(
    raw_events: list[dict], host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL
) -> list[ClassifiedEvent]:
    """
    Label every row via the local Ollama server. Returns [] immediately, no
    network call, if the server/model isn't reachable -- callers should treat
    that the same as "stage 2 unavailable, leave these rows uncertain" rather
    than crash.
    """
    if not is_available(host, model):
        logger.warning(
            "ollama_labeler: Ollama not reachable at %s or model %r not pulled -- "
            "skipping, no rows labeled. See labeling/README.md for the install/pull steps.",
            host, model,
        )
        return []

    results: list[ClassifiedEvent] = []
    for i, raw_event in enumerate(raw_events, start=1):
        labeled = _label_one(raw_event, host, model)
        if labeled is not None:
            results.append(labeled)
        if i % 50 == 0:
            logger.info("ollama_labeler: %d/%d processed", i, len(raw_events))

    logger.info("ollama_labeler: labeled %d/%d attempted rows", len(results), len(raw_events))
    return results
