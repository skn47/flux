"""
LLM-assisted labeler, key-gated on ANTHROPIC_API_KEY. NOT verified live in
this environment (no key is configured here) -- the only code path actually
exercised by `labeling/run_label.py` in this environment is the missing-key
skip path, same shape as `ingestion/newsapi.py`'s NEWSAPI_KEY handling.

**Skill note:** this environment does not have a local `claude-api` skill
installed (checked `~/.claude/skills`, plugin marketplaces, etc. -- none
found). Per the persona's trigger rule, this file was instead written after
fetching the current guidance directly from Anthropic's public
`anthropics/skills` repo (`skills/claude-api/shared/models.md` and
`skills/claude-api/SKILL.md`, fetched 2026-07-19) and confirmed against the
actually-installed `anthropic` Python SDK (v0.117.0) in this venv:
  - Structured output uses `client.messages.parse(..., output_format=<pydantic
    model>)`, which validates the response and exposes it at
    `response.parsed_output` -- the modern replacement for hand-rolled
    tool-use JSON parsing. Confirmed against
    `anthropic.types.parsed_message.ParsedMessage` in the installed package.
  - The fetched skill guidance defaults to `claude-opus-4-8` for general
    assistant use. **Deliberate deviation, documented, not silent:** this
    labeler instead defaults to `claude-haiku-4-5`, because this is a
    cost-sensitive, high-volume *batch* classification pipeline (the plan
    explicitly caps batch size "for cost/rate control"), not an interactive
    session -- Haiku is Anthropic's fastest/cheapest tier and is adequate for
    a short structured-classification task on a news snippet. The model is a
    module-level constant (`MODEL_ID`), trivially overridden if a reviewer
    disagrees with this tradeoff.

Batch size is capped by MAX_EVENTS_PER_RUN (cost/rate control, per the plan).
One API call per event (not one call for a whole batch) -- simpler to reason
about and to make idempotent/retriable per-row; a genuinely higher-throughput
design (packing many events into one prompt) is a reasonable future
optimization, not implemented here since this path is unverified anyway.
"""

from __future__ import annotations

import logging
import os
from typing import Literal, Optional

from config.mvp_scope import COUNTRIES, SECTORS, TRACKED_TICKERS
from src.ingestion.env import load_env_file
from src.labeling.schema import EVENT_TYPES, ClassifiedEvent, LABEL_SOURCES, now_iso, severity_band

logger = logging.getLogger("labeling.llm_labeler")

LABELER_VERSION = "llm_labeler_v1"
MODEL_ID = os.environ.get("ANTHROPIC_LABELER_MODEL", "claude-haiku-4-5")
MAX_EVENTS_PER_RUN = 50  # cost/rate control, see module docstring
REQUEST_MAX_TOKENS = 1024

_COUNTRY_NAMES = [c["name"] for c in COUNTRIES]
# S0 REWORK: was a single hardcoded sector name; now every registered sector's
# name (config/mvp_scope.py::SECTORS) so this prompt stays correct as
# S1/S2/S3 add financials/energy/pharma_biotech without another edit here.
_SECTOR_NAMES = list(SECTORS.keys())

_SYSTEM_PROMPT = f"""You are a careful financial-event classification assistant for a \
multi-sector flux-monitoring pipeline (tracked sectors: {_SECTOR_NAMES}). Classify the given \
news article title/text into this CLOSED taxonomy -- do not invent categories:

{chr(10).join(f"- {t}" for t in EVENT_TYPES)}

Category definitions:
- trade_export_control: export-control policy, entity-list actions, tariffs, sanctions, chip-specific trade restrictions.
- geopolitical_military_tension: military/security posturing or conflict risk to the corridor (cross-strait, Korea peninsula, US-China friction) without a physical disruption yet.
- supply_chain_fab_disruption: actual operational disruption (fab outage, shortage, factory fire, cyberattack, logistics disruption) -- the outcome, not the hazard.
- natural_disaster: the physical hazard itself (earthquake, typhoon, flood, drought).
- regulatory_approval_decision: a regulator's own decision about whether/how a SPECIFIC drug/biologic/device may be marketed -- FDA approval, rejection (Complete Response Letter), clinical trial (Phase 2/3) readouts, advisory-committee votes, or special designations (Breakthrough Therapy, Priority Review, Orphan Drug, Fast Track, RMAT). Distinct from regulatory_subsidy_action (below): this is about ONE product's own regulatory/clinical-evidentiary status, not a firm-level subsidy/antitrust/compliance action. Distinct from corporate_strategic's "product launch": the regulator's decision on whether a product CAN be sold comes first and is external; the company's own commercial launch strategy for an already-approved product is corporate_strategic.
- regulatory_subsidy_action: subsidies/grants (e.g. CHIPS Act), antitrust, non-trade regulatory rulings.
- macroeconomic_conditions: rates, inflation, GDP, general demand/inventory-cycle commentary not tied to a specific policy/company action.
- corporate_strategic: M&A, capacity announcements, earnings/guidance fluxs, executive changes, product launches.
- unclassified: use this if nothing above clears your own confidence floor -- do not force a guess.

Severity is an anchored 1-5 band (1=negligible/routine, 2=minor/contained, 3=moderate concrete-but-contained action, 4=major enacted/binding action, 5=severe/existential-crisis-level e.g. war, invasion, total embargo). severity_score is a continuous 0.0-1.0 estimate consistent with that band. polarity is -1.0 (very adverse to the corridor) to +1.0 (favorable), or null if you cannot tell direction.

Only report countries from this fixed set if genuinely relevant: {_COUNTRY_NAMES}. Only report sector as one of {_SECTOR_NAMES} if the article is genuinely about that industry, else null. Only report companies as tickers from this fixed set if genuinely mentioned/implicated: {TRACKED_TICKERS}. Never fabricate an entity that isn't actually supported by the text.

confidence (0.0-1.0) must honestly reflect your own uncertainty -- do not default to a high number. reasoning must be a short (1-2 sentence) justification."""


try:
    from pydantic import BaseModel, Field

    class _EventLabelOutput(BaseModel):
        event_type: Literal[tuple(EVENT_TYPES)]  # type: ignore[valid-type]
        severity: int = Field(ge=1, le=5)
        severity_score: float = Field(ge=0.0, le=1.0)
        polarity: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
        countries: list[str] = Field(default_factory=list)
        sector: Optional[str] = None
        companies: list[str] = Field(default_factory=list)
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str

    _PYDANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover - defensive, pydantic ships with anthropic sdk
    _EventLabelOutput = None  # type: ignore[assignment]
    _PYDANTIC_AVAILABLE = False


def _get_api_key() -> Optional[str]:
    load_env_file()  # no-op if .env doesn't exist; never overrides a real env var
    return os.environ.get("ANTHROPIC_API_KEY") or None


def _build_client(api_key: str):
    import anthropic  # lazy import: package may not be needed/installed in all environments

    return anthropic.Anthropic(api_key=api_key)


def _label_one(client, raw_event: dict) -> Optional[ClassifiedEvent]:
    title = raw_event.get("title") or ""
    text = raw_event.get("text") or ""
    article = f"Title: {title}\n\nText: {text}".strip()
    if not article or article == "Title: \n\nText:":
        logger.warning(
            "llm_labeler: skipping event_id=%s, no title/text available",
            raw_event.get("id"),
        )
        return None

    try:
        response = client.messages.parse(
            model=MODEL_ID,
            max_tokens=REQUEST_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": article}],
            output_format=_EventLabelOutput,
        )
        parsed = response.parsed_output
    except Exception as exc:  # noqa: BLE001 - one bad row must not kill the batch
        logger.warning(
            "llm_labeler: request failed for event_id=%s: %s", raw_event.get("id"), exc
        )
        return None

    if parsed is None:
        logger.warning(
            "llm_labeler: model returned no parsable structured output for event_id=%s",
            raw_event.get("id"),
        )
        return None

    return ClassifiedEvent(
        event_id=raw_event["id"],
        label_source="llm_assisted",
        event_type=parsed.event_type,
        severity=parsed.severity,
        severity_score=parsed.severity_score,
        confidence=parsed.confidence,
        polarity=parsed.polarity,
        countries=parsed.countries,
        sector=parsed.sector,
        companies=parsed.companies,
        reasoning=parsed.reasoning,
        labeler_version=f"{LABELER_VERSION}:{MODEL_ID}",
        labeled_at=now_iso(),
    )


def label_batch(raw_events: list[dict]) -> list[ClassifiedEvent]:
    """
    Label up to MAX_EVENTS_PER_RUN rows via the Anthropic API. Returns []
    immediately, without any network call, if ANTHROPIC_API_KEY is unset --
    this is the only path exercised in this environment.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.warning(
            "llm_labeler: ANTHROPIC_API_KEY not set (checked environment and .env) -- "
            "skipping LLM-assisted labeler, no network call made."
        )
        return []

    if not _PYDANTIC_AVAILABLE:
        logger.warning(
            "llm_labeler: pydantic is unavailable (should ship with the anthropic package) -- "
            "cannot build a structured-output schema, skipping."
        )
        return []

    try:
        client = _build_client(api_key)
    except ImportError as exc:
        logger.warning(
            "llm_labeler: 'anthropic' package not installed (%s) -- skipping, no network call made.",
            exc,
        )
        return []

    batch = raw_events[:MAX_EVENTS_PER_RUN]
    if len(raw_events) > MAX_EVENTS_PER_RUN:
        logger.info(
            "llm_labeler: %d unlabeled rows found, capping this run to %d (MAX_EVENTS_PER_RUN)",
            len(raw_events),
            MAX_EVENTS_PER_RUN,
        )

    results: list[ClassifiedEvent] = []
    for raw_event in batch:
        labeled = _label_one(client, raw_event)
        if labeled is not None:
            results.append(labeled)

    logger.info("llm_labeler: labeled %d/%d attempted rows", len(results), len(batch))
    return results
