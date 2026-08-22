"""
GDELT 2.0 Events connector. No API key needed.

Pipeline:
  1. GET http://data.gdeltproject.org/gdeltv2/lastupdate.txt to find the URL
     of the most recent 15-minute Events export zip.
  2. Download that zip, read the single tab-separated CSV inside it in
     memory (zipfile + io + csv, no pandas).
  3. Parse each row against the verified GDELT 2.0 Events column layout
     (61 columns -- confirmed against a live export file on 2026-07-18 and
     cross-checked against GDELT's own "THE GDELT EVENT DATABASE DATA FORMAT
     CODEBOOK V2.0" at
     http://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf).
  4. Keep only rows where Actor1CountryCode or Actor2CountryCode matches one
     of the MVP countries, to keep volume manageable.

Important nuance verified against GDELT's own reference files (not guessed):
GDELT uses TWO different country-code vocabularies in the same table.
  - Actor1CountryCode / Actor2CountryCode (columns 7 and 17) are 3-letter
    CAMEO codes (USA / TWN / KOR), NOT FIPS 10-4.
    Source: https://www.gdeltproject.org/data/lookups/CAMEO.country.txt
  - Actor1Geo_CountryCode / Actor2Geo_CountryCode / ActionGeo_CountryCode
    (columns 37, 45, 53) are 2-letter FIPS 10-4 codes (US / TW / KS).
    Source: https://www.gdeltproject.org/data/lookups/FIPS.country.txt
This connector filters on Actor1CountryCode/Actor2CountryCode (per the MVP
filter spec), so it matches against `CAMEO_COUNTRY_CODES`. The FIPS codes are
still carried in config.mvp_scope for any future connector/filter that reads
the Geo_CountryCode fields instead.

This connector does NOT classify events (no sector/severity judgment) --
that's Phase B/C. It ingests faithfully and stores the full matched row in
raw_metadata.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from datetime import datetime, timezone
from typing import Optional

import requests

from config.mvp_scope import CAMEO_COUNTRY_CODES
from src.ingestion.schema import RawEvent, make_event_id

logger = logging.getLogger("ingestion.gdelt")

SOURCE_NAME = "gdelt"
LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
REQUEST_TIMEOUT = 60

# GDELT 2.0 Events column indices (0-based), verified against a live export
# file and the official V2.0 codebook. Total column count: 61.
COL_GLOBALEVENTID = 0
COL_SQLDATE = 1
COL_ACTOR1CODE = 5
COL_ACTOR1NAME = 6
COL_ACTOR1COUNTRYCODE = 7
COL_ACTOR2CODE = 15
COL_ACTOR2NAME = 16
COL_ACTOR2COUNTRYCODE = 17
COL_ISROOTEVENT = 25
COL_EVENTCODE = 26
COL_EVENTBASECODE = 27
COL_EVENTROOTCODE = 28
COL_QUADCLASS = 29
COL_GOLDSTEINSCALE = 30
COL_NUMMENTIONS = 31
COL_NUMSOURCES = 32
COL_NUMARTICLES = 33
COL_AVGTONE = 34
COL_ACTIONGEO_COUNTRYCODE = 53
COL_DATEADDED = 59
COL_SOURCEURL = 60
EXPECTED_NUM_COLUMNS = 61


class GdeltConnectorError(Exception):
    pass


def _find_latest_export_zip_url(session: requests.Session) -> str:
    resp = session.get(LASTUPDATE_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ")
        url = parts[-1]
        if url.endswith(".export.CSV.zip"):
            return url
    raise GdeltConnectorError(
        f"No '.export.CSV.zip' entry found in {LASTUPDATE_URL} response"
    )


def _parse_dateadded_to_iso(dateadded: str) -> Optional[str]:
    """DATEADDED is YYYYMMDDHHMMSS, UTC. Returns ISO 8601 UTC or None if unparseable."""
    try:
        dt = datetime.strptime(dateadded.strip(), "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
        return dt.isoformat()
    except (ValueError, AttributeError):
        return None


def _row_matches_mvp_countries(row: list[str]) -> bool:
    actor1 = row[COL_ACTOR1COUNTRYCODE].strip()
    actor2 = row[COL_ACTOR2COUNTRYCODE].strip()
    return actor1 in CAMEO_COUNTRY_CODES or actor2 in CAMEO_COUNTRY_CODES


def _row_to_raw_event(row: list[str], ingested_at: str) -> Optional[RawEvent]:
    global_event_id = row[COL_GLOBALEVENTID].strip() or None
    dateadded_raw = row[COL_DATEADDED].strip()
    published_at = _parse_dateadded_to_iso(dateadded_raw)
    if published_at is None:
        logger.warning(
            "gdelt: skipping GLOBALEVENTID=%s, unparseable DATEADDED=%r",
            global_event_id,
            dateadded_raw,
        )
        return None

    sourceurl_raw = row[COL_SOURCEURL].strip()
    url = sourceurl_raw if sourceurl_raw.lower().startswith("http") else None

    try:
        event_id = make_event_id(SOURCE_NAME, url, global_event_id)
    except ValueError:
        logger.warning(
            "gdelt: skipping row with no usable URL or GLOBALEVENTID: %r", row
        )
        return None

    def _f(idx: int) -> Optional[float]:
        v = row[idx].strip()
        try:
            return float(v) if v else None
        except ValueError:
            return None

    def _i(idx: int) -> Optional[int]:
        v = row[idx].strip()
        try:
            return int(v) if v else None
        except ValueError:
            return None

    raw_metadata = {
        "GLOBALEVENTID": global_event_id,
        "SQLDATE": row[COL_SQLDATE].strip() or None,
        "Actor1Code": row[COL_ACTOR1CODE].strip() or None,
        "Actor1Name": row[COL_ACTOR1NAME].strip() or None,
        "Actor1CountryCode": row[COL_ACTOR1COUNTRYCODE].strip() or None,
        "Actor2Code": row[COL_ACTOR2CODE].strip() or None,
        "Actor2Name": row[COL_ACTOR2NAME].strip() or None,
        "Actor2CountryCode": row[COL_ACTOR2COUNTRYCODE].strip() or None,
        "IsRootEvent": row[COL_ISROOTEVENT].strip() or None,
        "EventCode": row[COL_EVENTCODE].strip() or None,
        "EventBaseCode": row[COL_EVENTBASECODE].strip() or None,
        "EventRootCode": row[COL_EVENTROOTCODE].strip() or None,
        "QuadClass": _i(COL_QUADCLASS),
        "GoldsteinScale": _f(COL_GOLDSTEINSCALE),
        "NumMentions": _i(COL_NUMMENTIONS),
        "NumSources": _i(COL_NUMSOURCES),
        "NumArticles": _i(COL_NUMARTICLES),
        "AvgTone": _f(COL_AVGTONE),
        "ActionGeo_CountryCode": row[COL_ACTIONGEO_COUNTRYCODE].strip() or None,
        "SOURCEURL": sourceurl_raw or None,
        "DATEADDED": dateadded_raw or None,
    }

    return RawEvent(
        id=event_id,
        source=SOURCE_NAME,
        source_id=global_event_id,
        published_at=published_at,
        ingested_at=ingested_at,
        title=None,  # GDELT events are structured records, not articles
        text=None,
        url=url,
        raw_metadata=raw_metadata,
    )


def fetch() -> list[RawEvent]:
    """
    Fetch the latest GDELT 2.0 Events export, filter to MVP countries, and
    return the normalized RawEvent list. Raises on hard failures (network,
    zip format); skips and logs individual malformed rows rather than
    aborting the whole batch.
    """
    ingested_at = datetime.now(timezone.utc).isoformat()
    session = requests.Session()
    session.headers.update({"User-Agent": "flux-ingest/0.1 (Phase A GDELT connector)"})

    zip_url = _find_latest_export_zip_url(session)
    logger.info("gdelt: fetching %s", zip_url)
    resp = session.get(zip_url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    events: list[RawEvent] = []
    skipped_malformed = 0
    skipped_no_country_match = 0

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        if not names:
            raise GdeltConnectorError(f"{zip_url} contained no files")
        with zf.open(names[0]) as f:
            text_stream = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            reader = csv.reader(text_stream, delimiter="\t")
            for row in reader:
                if len(row) != EXPECTED_NUM_COLUMNS:
                    skipped_malformed += 1
                    continue
                if not _row_matches_mvp_countries(row):
                    skipped_no_country_match += 1
                    continue
                event = _row_to_raw_event(row, ingested_at)
                if event is not None:
                    events.append(event)

    logger.info(
        "gdelt: parsed export, kept %d MVP-country rows (skipped %d malformed, "
        "%d out-of-scope-country)",
        len(events),
        skipped_malformed,
        skipped_no_country_match,
    )
    return events
