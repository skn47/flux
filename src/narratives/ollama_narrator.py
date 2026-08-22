"""
Generates a short, plain-English "why this event matters to markets"
narrative per event, for the 3D causality-graph feature.

Mirrors labeling/ollama_labeler.py's call shape exactly: same local Ollama
server over plain HTTP, same is_available() reachability gate (returns []
rather than raising if Ollama isn't up or the model isn't pulled), same
JSON-schema-constrained `format`, same defensive parsing discipline.

Critically, this does NOT let the model invent a causal mechanism. The
prompt is grounded in real, already-audited facts only: the event's
headline/type/corridor, plus 1-2 real edge justification strings pulled from
propagation/graph.py's EDGES_BY_PAIR (the same `Edge.note` citations backing
progress/exposure_graph.md) for the tickers this event's exposure graph query
actually reaches. The model is explicitly told to explain using only the
given facts -- not to fabricate a mechanism, a dollar figure, or predict
price direction. This keeps every generated narrative traceable back to a
real, human-reviewed citation, the same discipline propagation/graph.py's own
module docstring establishes for edge weights (never a hash-derived or
arbitrary placeholder).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import requests

from src.labeling.schema import now_iso
from src.propagation.graph import EDGES_BY_PAIR, ExposureGraph

logger = logging.getLogger("narratives.ollama_narrator")

NARRATOR_VERSION = "ollama_narrator_v1"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b-instruct"
REQUEST_TIMEOUT = 60
REACHABILITY_TIMEOUT = 3

_MAX_GROUNDING_EDGES = 3  # cap on how many real Edge.note citations feed the prompt


@dataclass(frozen=True)
class EventNarrative:
    event_id: str
    narrative: str
    model_version: str
    generated_at: str


_SYSTEM_PROMPT = """You are a financial analyst writing a SHORT plain-English \
explanation of why a real-world news event could matter for markets, for a \
trading-terminal UI read by a general (non-analyst) audience.

You are given a news headline, its classified event type, the affected \
country corridor, and a short list of already-vetted analyst facts \
describing HOW that corridor's exposure transmits to specific companies. \
Using ONLY those given facts, write ONE to THREE sentences (under 60 words \
total) explaining, causally, why this event could matter to the tickers it \
reaches. Be concrete and specific to the given facts. Do NOT invent a \
mechanism, a dollar figure, or a statistic that isn't given. Do NOT predict \
whether the price will go up or down. Plain English, no jargon, no hedging \
filler like "it is possible that.\""""

_RESPONSE_FORMAT = {
    "type": "object",
    "properties": {"narrative": {"type": "string"}},
    "required": ["narrative"],
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


def _clean_note(note: str) -> str:
    """Strip the leading '§anchor: ' prefix for display/prompt use -- the
    anchor itself is only meaningful as a progress/exposure_graph.md cross-ref."""
    if note.startswith("§") and ": " in note:
        return note.split(": ", 1)[1]
    return note


def top_grounding(corridor: list[str], event_type: str, limit: int = _MAX_GROUNDING_EDGES) -> tuple[Optional[str], list[str]]:
    """
    Real, human-reviewed edge citations for this event's SINGLE highest-impact
    exposure path only -- never fabricated, and never pooled across multiple
    tickers. Returns (ticker, notes) or (None, []) if nothing reachable.

    Deliberately restricted to one ticker's path, not the union of several
    top-impact tickers': an earlier version pooled notes from every top
    ticker, and when a corridor spans multiple unrelated countries (observed
    live: a Taiwan-arms headline whose classifier also attached "Russia" to
    its corridor), the pooled facts described entirely separate transmission
    chains (Taiwan->TSM manufacturing vs. Russia->energy-sector exposure).
    The 3B model then invented a connection between them that doesn't exist
    ("Russia provides >90% of its military capacity" -- actually TSM's own
    Taiwan manufacturing fact, misattributed). Restricting to one coherent
    path removes that failure mode by construction rather than trying to
    prompt around it a second time -- same discipline labeling/ollama_labeler.py
    already established for its own 3B-model failure modes.
    """
    if not corridor:
        return None, []
    exposures = ExposureGraph().query(set(corridor), event_type)
    if not exposures:
        return None, []
    ticker, exp = max(exposures.items(), key=lambda kv: kv[1].impact)

    notes: list[str] = []
    seen: set[str] = set()
    for src, dst in zip(exp.path, exp.path[1:]):
        for edge in EDGES_BY_PAIR.get((src, dst), []):
            cleaned = _clean_note(edge.note)
            if cleaned not in seen:
                seen.add(cleaned)
                notes.append(cleaned)
            if len(notes) >= limit:
                return ticker, notes
    return ticker, notes


def _user_prompt(title: str, event_type: str, ticker: Optional[str], notes: list[str]) -> str:
    lines = [
        f"Headline: {title}",
        f"Event type: {event_type}",
    ]
    if ticker:
        lines.append(f"Most-exposed tracked ticker: {ticker}")
        lines.append(f"Vetted facts about how this event's exposure reaches {ticker}:")
        if notes:
            lines += [f"- {n}" for n in notes]
        else:
            lines.append("- (no specific transmission facts on record)")
    else:
        lines.append("(no tracked ticker exposure on record for this event's corridor)")
    return "\n".join(lines)


def generate_one(
    event_id: str,
    title: str,
    event_type: str,
    corridor: list[str],
    host: str = OLLAMA_HOST,
    model: str = OLLAMA_MODEL,
) -> Optional[EventNarrative]:
    title = (title or "").strip()
    if not title:
        logger.warning("ollama_narrator: skipping event_id=%s, no title", event_id)
        return None

    ticker, notes = top_grounding(corridor, event_type)
    user_prompt = _user_prompt(title, event_type, ticker, notes)

    try:
        resp = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "format": _RESPONSE_FORMAT,
                "stream": False,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        logger.warning("ollama_narrator: request failed for event_id=%s: %s", event_id, exc)
        return None

    try:
        parsed = json.loads(content)
    except ValueError:
        logger.warning(
            "ollama_narrator: model returned non-JSON for event_id=%s: %r", event_id, content[:200],
        )
        return None

    narrative = parsed.get("narrative")
    if not isinstance(narrative, str) or not narrative.strip():
        logger.warning("ollama_narrator: model returned empty narrative for event_id=%s", event_id)
        return None

    return EventNarrative(
        event_id=event_id,
        narrative=narrative.strip(),
        model_version=f"{NARRATOR_VERSION}:{model}",
        generated_at=now_iso(),
    )


def generate_batch(
    candidates: list[dict], host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL
) -> list[EventNarrative]:
    """
    candidates: [{event_id, title, event_type, corridor: list[str]}, ...]
    Returns [] immediately, no network call, if Ollama isn't reachable --
    same graceful-degradation discipline as labeling/ollama_labeler.py.
    """
    if not is_available(host, model):
        logger.warning(
            "ollama_narrator: Ollama not reachable at %s or model %r not pulled -- "
            "skipping, no narratives generated.", host, model,
        )
        return []

    results: list[EventNarrative] = []
    for i, c in enumerate(candidates, start=1):
        narrated = generate_one(c["event_id"], c["title"], c["event_type"], c.get("corridor") or [], host, model)
        if narrated is not None:
            results.append(narrated)
        if i % 50 == 0:
            logger.info("ollama_narrator: %d/%d processed", i, len(candidates))

    logger.info("ollama_narrator: narrated %d/%d attempted rows", len(results), len(candidates))
    return results
