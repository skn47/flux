"""
Stage 1 (rule-based) content sanity-check for `gdelt_derived` events already
labeled `geopolitical_military_tension`.

Why this exists: `labeling/gdelt_labeler.py` derives `event_type` purely from
GDELT's `QuadClass` (3/4 -> geopolitical_military_tension) with zero content
signal -- CAMEO's root code 19 ("Fight") covers armed conflict, domestic
violence, and street crime with the same verb-level codes, so a domestic
murder-suicide story gets the identical treatment as a real military clash
(see progress/event_taxonomy.md's "Flagged ambiguity" entry for this discovery
and the concrete examples). This module gives that specific label a second,
content-based look using the real headline text scraped by
ingestion/article_titles.py -- something the original labeler never had.

This is a bootstrap sanity-check, not a production classifier (same posture
as labeling/rule_labeler.py's own docstring) -- its output is spot-checked by
hand (labeling/README.md), never presented as ground truth. It is
deliberately conservative: CONFIRM/REJECT only fire on a one-sided keyword
hit; anything ambiguous (both lists hit, or neither) falls through to
UNCERTAIN for a real (local-LLM) read in labeling/ollama_labeler.py, because
a gdelt_derived label started with *zero* content validation -- "no evidence
either way" must not default to trusting the original QuadClass guess.
"""

from __future__ import annotations

from typing import Literal

from src.labeling.rule_labeler import EVENT_TYPE_KEYWORDS, _contains_keyword

Verdict = Literal["confirm", "reject", "uncertain"]

# Two of rule_labeler.py's own keywords are excluded here: "military" (bare)
# and "north korea" (a bare country name). Both are reasonable signals in
# rule_labeler.py's original context -- full article text, where a generic
# hit is diluted by hundreds of surrounding words -- but in a short headline a
# single generic hit is often the ONLY signal, and it's too weak on its own:
# a live-corpus spot-check (labeling/README.md) found headlines like "North
# Korea partially reopens doors to tourists" and "North Korea reports 'fierce'
# heat as peninsula bakes" auto-confirmed as geopolitical_military_tension on
# nothing but the country name. Dropping both sends that whole bucket (153 of
# 677 confirm-stage1 titles, in the same spot-check) to Stage 2 for a real
# semantic read instead of a blind keyword auto-confirm -- affordable, since
# Stage 2 has been the reliably accurate part of this pipeline throughout.
_TOO_GENERIC_FOR_HEADLINES = {"military", "north korea"}

# The existing rule_labeler.py geopolitical_military_tension list (written
# for full article text, RSS/NewsAPI, minus the two terms above) plus a small
# set of additions tuned for short GDELT-style headlines during this module's
# design (validated read-only against the live corpus's 4,378 titled
# gdelt_derived geopolitical_military_tension rows before being committed
# here -- see progress/event_taxonomy.md).
GEO_KEYWORDS: list[str] = [
    k for k in EVENT_TYPE_KEYWORDS["geopolitical_military_tension"]
    if k not in _TOO_GENERIC_FOR_HEADLINES
] + [
    "sanction", "sanctions", "ceasefire", "airstrike", "air strike",
    "nuclear", "artillery", "combat", "insurgent", "militant", "rebel",
    "border clash", "annex", "sovereignty",
]

# New for this module -- headline patterns that indicate the story is
# domestic crime, an accident, entertainment/consumer content, or a routine
# police-blotter/civil matter, i.e. NOT geopolitical/military tension despite
# GDELT's QuadClass=4 coding. Deliberately specific multi-word phrases where
# a bare word would risk colliding with genuine conflict reporting (e.g.
# "shooting" alone is not listed -- "police arrest"/"deadly shooting in" is,
# since those phrasings are domestic-crime-report idioms).
REJECT_KEYWORDS: list[str] = [
    # domestic crime / interpersonal violence
    "murder-suicide", "domestic violence", "her husband", "her mother",
    "his wife", "custody", "stabbing", "stabbed", "robbery", "burglary",
    "dui", "sentenced", "arrested", "carjacking",
    # accidents
    "head-on crash", "car crash", "traffic accident", "highway crash",
    "collision", "fatal crash",
    # entertainment / consumer / sports / celebrity
    "review", "smartphone", "galaxy z", "iphone", "olympics", "box office",
    "album", "celebrity",
    # routine local police-blotter crime
    "police say", "police said", "police arrest", "deadly shooting in",
    # civil/environmental/legal matters unrelated to conflict
    "endangered species", "lawsuit", "zoning",
    # domestic National Guard activity (disaster response / city patrols /
    # civil unrest) -- "troop"/"troops" is a real GEO_KEYWORDS signal for
    # actual foreign-conflict reporting (verified: 46/47 confirm_stage1 rows
    # matching only via troop/troops were genuine, e.g. Russia/Ukraine and
    # North Korea troop-deployment stories -- not blanket-excluded like
    # "military"/"north korea" were), but all 3 "national guard" hits found
    # in the live corpus were domestic U.S. incidents wrongly confirmed by
    # that same "troops" hit (e.g. "National Guard troops fatally shoot a man
    # in downtown Memphis"). Adding this phrase doesn't blunt "troops"
    # elsewhere -- it only flips a headline that also says "National Guard"
    # from a false "confirm" to "uncertain" (both lists hit), for a real
    # Stage 2 read instead of either a blind confirm or a blind reject.
    "national guard",
]


def classify_title(title: str) -> Verdict:
    """One-sided keyword hit only. Both lists is 'uncertain', not a tie-break
    -- ambiguity here should fall through to a real read, not a guess.

    Uses labeling.rule_labeler's word-boundary-aware matcher, not a bare
    substring check -- that project's own spot-check found real false
    positives from plain substring matching (e.g. "war" inside "Toward"),
    see labeling/README.md. GEO_KEYWORDS inherits "war" directly from
    rule_labeler's own list, so this module needs the same fix, not a
    reintroduction of the bug it already found and fixed once."""
    has_geo = any(_contains_keyword(title, k) for k in GEO_KEYWORDS)
    has_reject = any(_contains_keyword(title, k) for k in REJECT_KEYWORDS)
    if has_geo and not has_reject:
        return "confirm"
    if has_reject and not has_geo:
        return "reject"
    return "uncertain"
