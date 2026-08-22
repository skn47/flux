"""Best-effort real-headline scraper for GDELT-derived events.

GDELT is a structured event export with no headline field -- `raw_events.title`
is always NULL for `source='gdelt'` rows (see ingestion/gdelt.py, gdelt_backfill.py).
GDELT rows do reliably carry a `url` (its own SOURCEURL column) though, so this
module closes that gap downstream: given a batch of article URLs, fetch each
page and extract its real title, caching every result (success or failure) in
a `scraped_titles` table keyed by URL so the same URL is never fetched twice.

Consumed by flux_engine/timeseries.py::enrich_event_catalog_urls(), which is
the only caller -- this module knows nothing about event_catalog/clusters,
only "give me titles for these URLs".

Politeness, mirroring ingestion/gdelt_backfill.py's established conventions
(the only other connector in this codebase fetching many external URLs at
scale) rather than inventing new ones:
  - bounded-concurrency ThreadPoolExecutor with a bounded in-flight queue
    (same max_in_flight = workers*2 pattern as gdelt_backfill.py, for the
    same reason: caps peak memory independent of batch size)
  - a shared requests.Session with a descriptive, self-identifying
    User-Agent (never a spoofed browser UA -- these are arbitrary
    third-party news domains this project has no relationship with, unlike
    GDELT's own API or NewsAPI's licensed feed)
  - a per-domain robots.txt check before every fetch -- new for this module
    specifically because, unlike GDELT/NewsAPI (licensed structured APIs)
    or RSS (feeds explicitly published for syndication), this is the first
    place the project scrapes arbitrary third-party web pages
  - only the <title>/og:title text is ever extracted and stored -- never
    the article body
  - permanent negative caching: a URL that fails is recorded with its
    failure status and never retried automatically, so a dead link doesn't
    get re-fetched on every pipeline refresh (see run_timeseries.py, which
    rebuilds the whole event_catalog -- and therefore calls
    enrich_event_catalog_urls -- on every run, not incrementally)
"""

from __future__ import annotations

import logging
import sqlite3
import time
import urllib.robotparser
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("ingestion.article_titles")

DEFAULT_DB_PATH = "data/events.db"
REQUEST_TIMEOUT = 20
ROBOTS_TIMEOUT = 10
MAX_RETRIES = 2
RETRY_BACKOFF_BASE = 2.0  # seconds; sleep = base * 2**attempt
USER_AGENT = "flux-ingest/0.1 (OSINT feed headline enrichment)"

_SCRAPED_TITLES_SCHEMA = """
CREATE TABLE IF NOT EXISTS scraped_titles (
    url         TEXT PRIMARY KEY,
    title       TEXT,
    status      TEXT NOT NULL,
    scraped_at  TEXT NOT NULL
);
"""


@dataclass
class ScrapeResult:
    title: Optional[str]
    status: str  # "ok" | "no_title" | "robots_disallowed" | "non_html" | "timeout" | "http_error" | "error"


def _robots_allows(url: str, session: requests.Session, robots_cache: dict) -> bool:
    """Best-effort robots.txt check, cached per-domain. A missing/unfetchable
    robots.txt is treated as allow-all (the common, permissive default), same
    as urllib.robotparser's own behavior when it can't read a robots.txt."""
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"
    if domain in robots_cache:
        return robots_cache[domain].can_fetch(USER_AGENT, url)

    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{domain}/robots.txt")
    try:
        resp = session.get(f"{domain}/robots.txt", timeout=ROBOTS_TIMEOUT)
        if resp.status_code >= 400:
            rp.parse([])  # no robots.txt -> allow-all
        else:
            rp.parse(resp.text.splitlines())
    except requests.RequestException:
        rp.parse([])  # unreachable -> allow-all, don't block on a flaky robots fetch
    robots_cache[domain] = rp
    return rp.can_fetch(USER_AGENT, url)


def _extract_title(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip()
        if title:
            return title
    if soup.title and soup.title.get_text():
        title = soup.title.get_text().strip()
        if title:
            return title
    return None


def fetch_title(url: str, session: requests.Session, robots_cache: dict) -> ScrapeResult:
    """Fetch one URL and extract its title. Never raises -- every failure
    mode is captured as a status string."""
    if not _robots_allows(url, session, robots_cache):
        return ScrapeResult(None, "robots_disallowed")

    last_status = "error"
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            content_type = resp.headers.get("Content-Type", "")
            if resp.status_code >= 400:
                return ScrapeResult(None, "http_error")
            if "html" not in content_type.lower():
                return ScrapeResult(None, "non_html")
            title = _extract_title(resp.text)
            return ScrapeResult(title, "ok" if title else "no_title")
        except requests.Timeout:
            last_status = "timeout"
        except requests.RequestException:
            last_status = "http_error"
        except Exception:  # noqa: BLE001 - parsing/decoding surprises, retry then give up
            last_status = "error"
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
    return ScrapeResult(None, last_status)


def _already_cached(con: sqlite3.Connection, urls: list[str]) -> set[str]:
    con.executescript(_SCRAPED_TITLES_SCHEMA)
    if not urls:
        return set()
    placeholders = ",".join("?" for _ in urls)
    rows = con.execute(
        f"SELECT url FROM scraped_titles WHERE url IN ({placeholders})", urls
    ).fetchall()
    return {row[0] for row in rows}


def fetch_missing_titles(
    urls: list[str], db_path: str = DEFAULT_DB_PATH, workers: int = 5
) -> None:
    """For every URL not already present in `scraped_titles` (any status --
    permanent negative caching), fetch it and store the result. Bounded
    concurrency, same shape as ingestion/gdelt_backfill.py's day-fetch loop."""
    distinct_urls = sorted(set(urls))
    con = sqlite3.connect(db_path)
    try:
        cached = _already_cached(con, distinct_urls)
        pending = [u for u in distinct_urls if u not in cached]
        if not pending:
            return

        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        robots_cache: dict = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            max_in_flight = max(workers * 2, 1)
            urls_iter = iter(pending)
            in_flight: dict = {}

            def _submit_next() -> bool:
                try:
                    u = next(urls_iter)
                except StopIteration:
                    return False
                in_flight[pool.submit(fetch_title, u, session, robots_cache)] = u
                return True

            for _ in range(min(max_in_flight, len(pending))):
                _submit_next()

            processed = 0
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    u = in_flight.pop(fut)
                    result = fut.result()
                    con.execute(
                        """
                        INSERT OR REPLACE INTO scraped_titles (url, title, status, scraped_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (u, result.title, result.status, datetime.now(timezone.utc).isoformat()),
                    )
                    con.commit()
                    processed += 1
                    if processed % 100 == 0:
                        logger.info(
                            "article_titles: %d/%d URLs scraped", processed, len(pending)
                        )
                    _submit_next()
    finally:
        con.close()


def lookup_titles(urls: list[str], db_path: str = DEFAULT_DB_PATH) -> dict[str, str]:
    """Return {url: title} for URLs with a cached status='ok' result."""
    if not urls:
        return {}
    con = sqlite3.connect(db_path)
    try:
        con.executescript(_SCRAPED_TITLES_SCHEMA)
        placeholders = ",".join("?" for _ in urls)
        rows = con.execute(
            f"SELECT url, title FROM scraped_titles WHERE status='ok' AND url IN ({placeholders})",
            urls,
        ).fetchall()
        return {row[0]: row[1] for row in rows}
    finally:
        con.close()
