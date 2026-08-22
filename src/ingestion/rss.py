"""
RSS/Atom connector. No API key needed.

FEEDS below is the *result* of an empirical bake-off run on 2026-07-18: every
candidate feed was actually fetched with feedparser and checked for
`feed.bozo` (parse errors) and entry count before being kept. Candidates that
failed to parse, returned zero entries, or 404'd were dropped -- see the
"kept vs dropped" notes below and ingestion/README.md for the full table.

Kept (valid feed, >0 entries, topically relevant to tech/business/APAC):
  - TechCrunch                    -- general tech news
  - CNBC Technology                -- tech/markets
  - CNBC Top News                  -- broad business/markets
  - MarketWatch Top Stories        -- US markets
  - Taipei Times                   -- Taiwan
  - Nikkei Asia                    -- APAC business (Japan-based, pan-Asia coverage)
  - Yahoo Finance                  -- US markets
  - Investing.com Stock Market News -- markets
  - Seeking Alpha Market News      -- markets
  - SCMP Business                  -- Hong Kong/China business, APAC
  - Channel News Asia Business     -- Singapore/APAC business
  - Google News: "semiconductor" search -- the only sector-specific feed;
    kept despite being a generated search feed (not a newsroom's own feed)
    because it's the one source that's actually semiconductor-targeted.
    Flagged in README as best-effort/unofficial.

Dropped (technical failure at test time):
  - Reuters Business/Technology (feeds.reuters.com) -- DNS no longer resolves,
    Reuters retired these feedburner-era RSS feeds.
  - Korea Herald Business  -- feedparser bozo=True, 0 entries (bad content-type,
    entries not parseable).
  - Focus Taiwan (business + main) -- 404 / malformed XML (mismatched tag),
    0 entries.
  - Yonhap News (business + all)  -- 404, XML syntax error, 0 entries.
  - Korea JoongAng Daily -- 301 redirect landed on malformed XML (invalid
    token), 0 entries.
  - Korea Times Business -- resolved (301) but 0 entries; effectively dead feed.
  - Taiwan News -- 307 redirect landed on malformed XML, 0 entries.
  - MarketWatch "Real-time Headlines" (feeds.marketwatch.com) -- parsed, but
    dropped as redundant with MarketWatch Top Stories (same publisher) rather
    than adding a second brittle feedburner-style URL.

Dropped (parsed fine, but excluded for topical relevance, not failure):
  - The Verge, Ars Technica -- valid, well-formed feeds, but consumer-tech
    product coverage rather than business/markets/semiconductor-flux signal.
    Left out of the default list to keep signal-to-noise reasonable for a
    financial-flux pipeline; trivial to re-add if Phase B wants them.
"""

from __future__ import annotations

import logging
from calendar import timegm
from datetime import datetime, timezone
from typing import Optional

import feedparser

from src.ingestion.schema import RawEvent, make_event_id

logger = logging.getLogger("ingestion.rss")

SOURCE_NAME = "rss"
REQUEST_TIMEOUT = 30
USER_AGENT = "flux-ingest/0.1 (Phase A RSS connector)"

FEEDS: dict[str, str] = {
    "TechCrunch": "https://techcrunch.com/feed/",
    "CNBC Technology": "https://www.cnbc.com/id/19854910/device/rss/rss.html",
    "CNBC Top News": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "MarketWatch Top Stories": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Taipei Times": "https://www.taipeitimes.com/xml/index.rss",
    "Nikkei Asia": "https://asia.nikkei.com/rss/feed/nar",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Investing.com Stock Market News": "https://www.investing.com/rss/news_25.rss",
    "Seeking Alpha Market News": "https://seekingalpha.com/market_currents.xml",
    "SCMP Business": "https://www.scmp.com/rss/92/feed",
    "Channel News Asia Business": "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6511",
    "Google News: semiconductor": "https://news.google.com/rss/search?q=semiconductor&hl=en-US&gl=US&ceid=US:en",
}


def _struct_time_to_iso(struct_time) -> Optional[str]:
    if struct_time is None:
        return None
    try:
        dt = datetime.fromtimestamp(timegm(struct_time), tz=timezone.utc)
        return dt.isoformat()
    except (ValueError, OverflowError, TypeError):
        return None


def _entry_to_raw_event(
    entry, feed_name: str, feed_url: str, ingested_at: str
) -> Optional[RawEvent]:
    url = entry.get("link") or None
    source_id = entry.get("id") or entry.get("guid") or None

    published_at = _struct_time_to_iso(entry.get("published_parsed"))
    fallback_used = None
    if published_at is None:
        published_at = _struct_time_to_iso(entry.get("updated_parsed"))
        fallback_used = "updated_parsed" if published_at else None
    if published_at is None:
        # Genuinely no timestamp from the source -- fall back to ingestion
        # time, but say so explicitly rather than pretending it's real.
        published_at = ingested_at
        fallback_used = "ingested_at"

    try:
        event_id = make_event_id(SOURCE_NAME, url, source_id)
    except ValueError:
        logger.warning(
            "rss: skipping entry with no URL and no id from feed %r: %r",
            feed_name,
            entry.get("title"),
        )
        return None

    title = entry.get("title") or None
    text = entry.get("summary") or entry.get("description") or None
    author = entry.get("author") or None
    tags = [t.get("term") for t in entry.get("tags", []) if t.get("term")]

    raw_metadata = {
        "feed_name": feed_name,
        "feed_url": feed_url,
        "author": author,
        "tags": tags,
        "guid": entry.get("guid"),
    }
    if fallback_used:
        raw_metadata["published_at_fallback"] = fallback_used

    return RawEvent(
        id=event_id,
        source=SOURCE_NAME,
        source_id=source_id,
        published_at=published_at,
        ingested_at=ingested_at,
        title=title,
        text=text,
        url=url,
        raw_metadata=raw_metadata,
    )


def fetch() -> list[RawEvent]:
    """
    Fetch every feed in FEEDS. A single broken feed is logged and skipped --
    it does not abort the rest of the RSS connector, let alone other sources.
    """
    ingested_at = datetime.now(timezone.utc).isoformat()
    events: list[RawEvent] = []

    for feed_name, feed_url in FEEDS.items():
        try:
            parsed = feedparser.parse(
                feed_url,
                request_headers={"User-Agent": USER_AGENT},
            )
        except Exception as exc:  # noqa: BLE001 - one bad feed must not kill the run
            logger.warning("rss: failed to fetch feed %r (%s): %s", feed_name, feed_url, exc)
            continue

        if parsed.bozo and not parsed.entries:
            logger.warning(
                "rss: feed %r (%s) failed to parse (bozo=%s, exc=%s), skipping this run",
                feed_name,
                feed_url,
                parsed.bozo,
                getattr(parsed, "bozo_exception", None),
            )
            continue
        if parsed.bozo:
            logger.warning(
                "rss: feed %r (%s) parsed with warnings (bozo=%s), using %d entries anyway",
                feed_name,
                feed_url,
                getattr(parsed, "bozo_exception", None),
                len(parsed.entries),
            )

        if not parsed.entries:
            logger.warning("rss: feed %r (%s) returned zero entries", feed_name, feed_url)
            continue

        for entry in parsed.entries:
            event = _entry_to_raw_event(entry, feed_name, feed_url, ingested_at)
            if event is not None:
                events.append(event)

    logger.info("rss: collected %d entries across %d configured feeds", len(events), len(FEEDS))
    return events
