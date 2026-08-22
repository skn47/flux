"""
NewsAPI connector (https://newsapi.org/v2/everything). Key-gated.

Reads NEWSAPI_KEY from the environment (populated either by a real env var
or by ingestion/env.py's tiny .env loader, which the orchestrator calls
before running any connector). If the key is missing, this connector logs a
single clear warning and returns an empty list WITHOUT making any network
call and WITHOUT raising -- a missing optional API key is not a crash-worthy
condition.

Not exercised against the live API in this environment (no key available
here) -- the code path that matters for this environment is the missing-key
skip path, which the orchestrator run does exercise.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import requests

from config.mvp_scope import ALL_COMPANY_NAMES
from src.ingestion.env import load_env_file
from src.ingestion.schema import RawEvent, make_event_id

logger = logging.getLogger("ingestion.newsapi")

SOURCE_NAME = "newsapi"
ENDPOINT = "https://newsapi.org/v2/everything"
REQUEST_TIMEOUT = 30
PAGE_SIZE = 100


class NewsApiSkipped(Exception):
    """Raised internally to signal a clean, expected skip (no key present)."""


def _get_api_key() -> Optional[str]:
    load_env_file()  # no-op if .env doesn't exist; never overrides a real env var
    return os.environ.get("NEWSAPI_KEY") or None


def _article_to_raw_event(article: dict, query: str, ingested_at: str) -> Optional[RawEvent]:
    url = article.get("url") or None
    published_at = article.get("publishedAt") or None
    if published_at is None:
        # NewsAPI's schema always includes publishedAt for real articles; if
        # it's genuinely absent, don't fabricate a time -- fall back to
        # ingestion time and say so, same discipline as the RSS connector.
        published_at = ingested_at
        fallback_used = "ingested_at"
    else:
        fallback_used = None

    try:
        event_id = make_event_id(SOURCE_NAME, url, None)
    except ValueError:
        logger.warning("newsapi: skipping article with no URL: %r", article.get("title"))
        return None

    source_obj = article.get("source") or {}
    raw_metadata = {
        "query": query,
        "author": article.get("author"),
        "source_name": source_obj.get("name"),
        "source_id": source_obj.get("id"),
        "url_to_image": article.get("urlToImage"),
    }
    if fallback_used:
        raw_metadata["published_at_fallback"] = fallback_used

    return RawEvent(
        id=event_id,
        source=SOURCE_NAME,
        source_id=None,  # NewsAPI articles have no native id, only a URL
        published_at=published_at,
        ingested_at=ingested_at,
        title=article.get("title") or None,
        text=article.get("description") or article.get("content") or None,
        url=url,
        raw_metadata=raw_metadata,
    )


def fetch() -> list[RawEvent]:
    api_key = _get_api_key()
    if not api_key:
        logger.warning(
            "newsapi: NEWSAPI_KEY not set (checked environment and .env) -- "
            "skipping NewsAPI connector, no network call made."
        )
        return []

    ingested_at = datetime.now(timezone.utc).isoformat()
    session = requests.Session()
    session.headers.update({"X-Api-Key": api_key, "User-Agent": "flux-ingest/0.1"})

    # One OR-joined query across all tracked companies keeps this to a single
    # request per run rather than one per ticker, which matters for NewsAPI's
    # free-tier rate limits.
    query = " OR ".join(f'"{name}"' for name in ALL_COMPANY_NAMES)

    events: list[RawEvent] = []
    try:
        resp = session.get(
            ENDPOINT,
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": PAGE_SIZE,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        logger.error("newsapi: request failed: %s", exc)
        return []
    except ValueError as exc:
        logger.error("newsapi: could not parse JSON response: %s", exc)
        return []

    if payload.get("status") != "ok":
        logger.error("newsapi: API returned error status: %s", payload.get("message"))
        return []

    for article in payload.get("articles", []):
        event = _article_to_raw_event(article, query, ingested_at)
        if event is not None:
            events.append(event)

    logger.info("newsapi: fetched %d articles", len(events))
    return events
