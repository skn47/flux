"""
Multi-sector scope configuration for ingestion, labeling, and the exposure graph.

S0 (2026-07-23, see /home/h/.claude/plans/yes-kick-off-phase-stateful-trinket.md's
"Sector & ticker broadening (S0-S5)" status update) generalized this module from a
single hardcoded sector (semiconductors x United States/Taiwan/South Korea) into a
`SECTORS` registry, so a new sector (S1/S2/S3: financials, energy, pharma_biotech)
is a DATA addition here, not a structural change anywhere downstream.

Every module that only needs the FLAT aggregate lists (`TRACKED_TICKERS`,
`ALL_COMPANY_NAMES`, `FIPS_COUNTRY_CODES`, `CAMEO_COUNTRY_CODES`, `TRACKED_STOCKS`,
`COUNTRIES`) keeps working completely unchanged -- these are still exported at
module level with the exact same shape (list of plain dicts) as before this
refactor, now computed as a de-duplicated union across every registered sector
instead of being hand-written for one. Code that needs to know WHICH sector a
ticker belongs to (the labeler's sector-matching, the cross-sectional backtest's
future within-sector ranking) uses `sector_for_ticker()` / `SECTORS` directly.

Country codes
-------------
GDELT uses TWO different country-code vocabularies in the same Events table, and
conflating them silently would produce a connector that quietly filters on the
wrong field.

- FIPS 10-4 codes (2 characters) -- used in the `*Geo_CountryCode` fields
  (Actor1Geo_CountryCode, Actor2Geo_CountryCode, ActionGeo_CountryCode).
  Source: https://www.gdeltproject.org/data/lookups/FIPS.country.txt
- CAMEO country codes (3 characters) -- used in the `Actor1CountryCode` /
  `Actor2CountryCode` fields (these are the fields the GDELT connector filters
  on, since the task's MVP filter targets actor country affiliation).
  Source: https://www.gdeltproject.org/data/lookups/CAMEO.country.txt

Every sector below must verify its OWN countries against these two live reference
files before being added (not guessed from memory or reused from another sector's
codes without checking) -- see each sector's own inline citation/fetch-date note.
Both code vocabularies are kept per-country so a connector can filter on whichever
field it actually reads without re-deriving codes from memory.
"""

from __future__ import annotations

# --- Sector registry ---------------------------------------------------------
# Each sector: {"countries": [{"name","fips10_4","cameo"}, ...],
#               "stocks": [{"ticker","company_names": [...]}, ...]}
# Adding a sector here is the ENTIRE code change needed to bring its tickers
# into every downstream module (price/flux_engine/features/lstm/backtest all
# already loop over the flat TRACKED_TICKERS aggregate below) -- only the
# exposure graph (propagation/graph.py) and event-taxonomy-aware labeling
# (labeling/rule_labeler.py's SECTOR_KEYWORDS) need matching sector-specific
# entries of their own, since those genuinely encode sector-specific content.
SECTORS: dict[str, dict] = {
    "semiconductors": {
        # Verified against GDELT's own reference files, fetched 2026-07-18.
        # US -> United States, TW -> Taiwan, KS -> South Korea (FIPS 10-4;
        # deliberately not ISO "KR", which FIPS 10-4 does not use).
        "countries": [
            {"name": "United States", "fips10_4": "US", "cameo": "USA"},
            {"name": "Taiwan", "fips10_4": "TW", "cameo": "TWN"},
            {"name": "South Korea", "fips10_4": "KS", "cameo": "KOR"},
        ],
        "stocks": [
            {"ticker": "NVDA", "company_names": ["NVIDIA", "NVIDIA Corporation"]},
            {
                "ticker": "TSM",
                "company_names": [
                    "Taiwan Semiconductor Manufacturing Company",
                    "Taiwan Semiconductor",
                    "TSMC",
                ],
            },
            {"ticker": "AMD", "company_names": ["Advanced Micro Devices", "AMD"]},
            {"ticker": "ASML", "company_names": ["ASML Holding", "ASML"]},
            {"ticker": "INTC", "company_names": ["Intel", "Intel Corporation"]},
            # --- R3 (2026-07-23) additions -- see progress/ticker_universe_expansion.md
            # for the sector-membership selection criterion (written before any
            # candidate's returns were consulted) and the cited research behind
            # each pick.
            {"ticker": "MU", "company_names": ["Micron", "Micron Technology"]},
            {
                "ticker": "QCOM",
                "company_names": ["Qualcomm", "Qualcomm Incorporated", "QUALCOMM Incorporated"],
            },
            {"ticker": "AMAT", "company_names": ["Applied Materials"]},
        ],
    },
    "financials": {
        # S1 (2026-07-23) addition -- see progress/sector_expansion_financials.md for the
        # sector-membership selection criterion (written before any candidate's returns
        # were consulted) and the cited research behind each pick. Deliberately reuses
        # the EXACT same United States country dict as the "semiconductors" entry above
        # (_dedupe_countries below raises if the codes ever diverge) -- Financials is
        # scoped to US-domiciled/US-concentrated companies only, so no new country node
        # or GDELT FIPS/CAMEO corridor work is needed (unlike Energy's future S2).
        "countries": [
            {"name": "United States", "fips10_4": "US", "cameo": "USA"},
        ],
        "stocks": [
            {
                "ticker": "JPM",
                "company_names": ["JPMorgan Chase", "JPMorgan Chase & Co.", "JPMorgan"],
            },
            {
                "ticker": "SCHW",
                # Deliberately NOT including bare "Schwab" -- collides with Klaus Schwab
                # (WEF founder), a real, well-known false-positive risk; the two-word
                # "Charles Schwab" phrase avoids it, same discipline as the word-boundary
                # matching fix documented in labeling/rule_labeler.py.
                "company_names": ["Charles Schwab", "Charles Schwab Corporation"],
            },
            {"ticker": "ZION", "company_names": ["Zions Bancorporation", "Zions Bank"]},
            {
                "ticker": "PGR",
                # Deliberately NOT including bare "Progressive" -- a common generic word
                # (progressive politics/tax/era) with a much worse false-positive profile
                # than any existing ticker alias; same discipline as the SCHW note above.
                "company_names": ["Progressive Corporation", "The Progressive Corporation"],
            },
            {
                "ticker": "COF",
                "company_names": ["Capital One", "Capital One Financial Corporation"],
            },
        ],
    },
    "energy": {
        # S2 (2026-07-23) addition -- see progress/sector_expansion_energy.md for the sector-
        # membership selection criterion (written before any candidate's returns were
        # consulted), the GDELT FIPS/CAMEO code verification, and the cited research behind
        # each pick. Unlike financials (S1), Energy's real geopolitical exposure genuinely
        # runs through countries OUTSIDE the semiconductor corridor -- Saudi Arabia (OPEC+
        # production/swing-capacity risk) and Russia (sanctions/supply-disruption risk) --
        # so this sector required new FIPS/CAMEO codes, independently fetched and verified
        # against GDELT's own live reference files (not assumed/guessed from memory), same
        # discipline as this module's original 3 codes. `United States` is reused verbatim
        # (identical dict to the semiconductors/financials entries above) since
        # `_dedupe_countries` below requires an exact match to silently reuse an entry.
        #
        # Russia is FIPS 10-4 "RS", NOT ISO 3166-1's "RU" -- a second, independently-hit
        # example of the exact same ISO-vs-FIPS trap this module's docstring already flags
        # for South Korea ("KS" not "KR"). Saudi Arabia's FIPS code ("SA") happens to match
        # its ISO code, but was verified with the same rigor rather than assumed safe because
        # it "looked right". Verified live 2026-07-23 via direct `curl` of:
        #   https://www.gdeltproject.org/data/lookups/FIPS.country.txt   (RS Russia, SA Saudi Arabia)
        #   https://www.gdeltproject.org/data/lookups/CAMEO.country.txt  (RUS Russia, SAU Saudi Arabia)
        #
        # ⚠ Known coverage gap, honestly flagged (see progress/sector_expansion_energy.md §2c):
        # ingestion/gdelt.py's and ingestion/gdelt_backfill.py's country filter historically
        # required Actor1 OR Actor2 to be in the (pre-S2) 3-country CAMEO list, so the
        # existing raw_events corpus already contains real Russia/Saudi-Arabia-tagged rows
        # (49,823 / 6,239 respectively, as of 2026-07-23) but ONLY where the other actor is
        # USA/TWN/KOR (96-98% paired with USA specifically) -- it structurally cannot contain
        # Russia-Europe gas-pipeline events, Russia-only domestic events, or Saudi-Arabia-only
        # OPEC+ announcements with no US/Taiwan/Korea angle, since those would never have
        # passed the pre-S2 filter. This config addition makes FUTURE ingestion/backfill runs
        # capture those directly; it does not retroactively backfill the gap in already-elapsed
        # history. No new historical GDELT backfill was launched to fill this -- flagged for a
        # separate decision, per the plan's explicit instruction not to absorb that larger,
        # riskier operation into this sector-addition pass.
        "countries": [
            {"name": "United States", "fips10_4": "US", "cameo": "USA"},
            {"name": "Saudi Arabia", "fips10_4": "SA", "cameo": "SAU"},
            {"name": "Russia", "fips10_4": "RS", "cameo": "RUS"},
        ],
        "stocks": [
            {"ticker": "XOM", "company_names": ["ExxonMobil", "Exxon Mobil", "Exxon Mobil Corporation"]},
            {
                # Deliberately NOT including bare "Occidental" -- a real generic-English-word
                # collision risk (occidental = "western", a common adjective in unrelated
                # contexts), same discipline as the SCHW/PGR word-collision-avoidance notes in
                # the financials sector above. "Occidental Petroleum" is specific and safe.
                "ticker": "OXY",
                "company_names": ["Occidental Petroleum", "Occidental Petroleum Corporation"],
            },
            {"ticker": "EOG", "company_names": ["EOG Resources"]},
            {"ticker": "LNG", "company_names": ["Cheniere Energy", "Cheniere"]},
            {"ticker": "VLO", "company_names": ["Valero Energy", "Valero Energy Corporation"]},
        ],
    },
    "pharma_biotech": {
        # S3 (2026-07-23) addition -- see progress/sector_expansion_pharma.md for the
        # sector-membership selection criterion (written before any candidate's returns
        # were consulted) and the cited research (10-Ks/investor materials) behind each
        # pick. Deliberately reuses the EXACT same United States country dict as the
        # semiconductors/financials/energy entries above (_dedupe_countries below raises
        # if the codes ever diverge) -- a deliberate scope decision made BEFORE S3
        # started, NOT an oversight: adding any new country node retriggers the broad
        # cross-sector flux_score shift discovered during the Saudi Arabia/Russia
        # relabel (flux_engine/formula.py::_cluster_key()'s corridor bucketing is
        # dynamically derived from every country node in propagation/graph.py::NODES).
        # Pharma's real EU/EMA regulatory exposure is knowingly left out of scope as a
        # result -- see progress/sector_expansion_pharma.md's scope-decision note.
        "countries": [
            {"name": "United States", "fips10_4": "US", "cameo": "USA"},
        ],
        "stocks": [
            {"ticker": "PFE", "company_names": ["Pfizer", "Pfizer Inc."]},
            {
                # Deliberately using the full "Vertex Pharmaceuticals" phrase, NOT bare
                # "Vertex" -- a real generic-English-word collision risk ("vertex" as a
                # geometry/common-usage term) plus a real distinct-company collision risk
                # (Vertex Inc., NASDAQ: VERX, a tax-software company), same discipline as
                # the SCHW/PGR/OXY word-collision-avoidance notes in the financials/energy
                # sectors above.
                "ticker": "VRTX",
                "company_names": ["Vertex Pharmaceuticals", "Vertex Pharmaceuticals Incorporated"],
            },
            {
                "ticker": "BMRN",
                "company_names": ["BioMarin", "BioMarin Pharmaceutical", "BioMarin Pharmaceutical Inc."],
            },
            {"ticker": "MRNA", "company_names": ["Moderna", "Moderna, Inc."]},
            {
                "ticker": "REGN",
                "company_names": ["Regeneron", "Regeneron Pharmaceuticals", "Regeneron Pharmaceuticals Inc."],
            },
        ],
    },
}

# --- Derived aggregates (union across all registered sectors) ---------------
# Same flat shapes as before this refactor (list of plain dicts / list of str)
# -- every existing `from config.mvp_scope import X` call site keeps working
# unmodified. De-duplicated by `name`/`ticker` so a country or ticker shared by
# two sectors (not the case today, but a real future possibility) doesn't
# produce duplicate rows.


def _dedupe_countries(sectors: dict[str, dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for sector in sectors.values():
        for c in sector["countries"]:
            if c["name"] in seen and seen[c["name"]] != c:
                raise ValueError(
                    f"country {c['name']!r} registered with conflicting codes "
                    f"across sectors: {seen[c['name']]} vs {c}"
                )
            seen[c["name"]] = c
    return list(seen.values())


def _dedupe_stocks(sectors: dict[str, dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for sector in sectors.values():
        for s in sector["stocks"]:
            if s["ticker"] in seen:
                raise ValueError(f"ticker {s['ticker']!r} registered in more than one sector")
            seen[s["ticker"]] = s
    return list(seen.values())


COUNTRIES = _dedupe_countries(SECTORS)
TRACKED_STOCKS = _dedupe_stocks(SECTORS)

FIPS_COUNTRY_CODES = [c["fips10_4"] for c in COUNTRIES]
CAMEO_COUNTRY_CODES = [c["cameo"] for c in COUNTRIES]
TRACKED_TICKERS = [s["ticker"] for s in TRACKED_STOCKS]
ALL_COMPANY_NAMES = [name for stock in TRACKED_STOCKS for name in stock["company_names"]]

# ticker -> sector name, e.g. "NVDA" -> "semiconductors". The one genuinely NEW
# capability this refactor adds: nothing before S0 could answer "which sector
# is this ticker in" because there was only ever one sector to be in.
TICKER_TO_SECTOR: dict[str, str] = {
    stock["ticker"]: sector_name
    for sector_name, sector in SECTORS.items()
    for stock in sector["stocks"]
}


def sector_for_ticker(ticker: str) -> str | None:
    """Sector name for a tracked ticker, or None if `ticker` isn't tracked."""
    return TICKER_TO_SECTOR.get(ticker)


# --- Backward-compat single-sector alias -------------------------------------
# S0 REWORK: `SECTOR` (singular) no longer means anything once more than one
# sector is registered -- kept ONLY while `SECTORS` has exactly one entry, and
# intentionally raises (rather than silently picking one) the moment a second
# sector is added, so any remaining `from config.mvp_scope import SECTOR`
# call site is forced to be updated to sector-aware logic instead of silently
# mislabeling every non-semiconductor article as "semiconductors". As of this
# refactor, `labeling/rule_labeler.py` and `labeling/llm_labeler.py` (the only
# two importers) have already been updated to not need this constant -- it
# remains only as an explicit trip-wire against a future regression.
if len(SECTORS) == 1:
    SECTOR: str = next(iter(SECTORS))
else:
    def __getattr__(name: str):
        if name == "SECTOR":
            raise ImportError(
                "config.mvp_scope.SECTOR is undefined once more than one sector "
                "is registered -- use sector_for_ticker() or SECTORS directly."
            )
        raise AttributeError(name)
