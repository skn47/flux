"""
Shared event schema for Phase A ingestion.

Every connector (GDELT, RSS, NewsAPI, ...) normalizes its native records into
a single `RawEvent`. Source-specific fields NEVER leak into the shared
top-level fields -- they go into `raw_metadata` as a dict. This keeps
downstream code (Phase B/C: NLP classification, propagation graph, flux
scoring) agnostic to which source an event came from.

Canonical timestamp discipline:
  - `published_at` is the source's own event/publish time whenever the source
    provides one, converted to ISO 8601 UTC. This is what downstream phases
    should treat as "when this happened in the world."
  - `ingested_at` is when *this* ingestion run pulled the record. It is
    always set, always UTC, and must never be used as a stand-in for
    `published_at`.
  - If a source genuinely provides no publish timestamp, connectors fall back
    to `ingested_at` for `published_at` but MUST record that fact in
    `raw_metadata` (e.g. `{"published_at_fallback": "ingested_at"}`) so
    nothing is silently backfilled/fabricated.

Deduplication:
  `id` is a stable sha256 hex digest of `source + "|" + dedup_key`, where
  `dedup_key` is the normalized URL when available, otherwise the source's
  native `source_id`. Same input -> same id, always -- this is what makes
  re-running ingestion idempotent (INSERT OR IGNORE keyed on `id`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional
from urllib.parse import urlsplit, urlunsplit


def normalize_url(url: Optional[str]) -> Optional[str]:
    """
    Normalize a URL for stable deduplication: lowercase scheme/host, strip
    fragment, strip a trailing slash. Deliberately conservative -- we do NOT
    strip query strings, since tracking params sometimes double as the only
    thing distinguishing two genuinely different articles on lazy CMSs. This
    is a dedup-key concern, not a "canonicalize for display" concern.
    """
    if not url:
        return None
    url = url.strip()
    if not url:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    scheme = (parts.scheme or "http").lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    normalized = urlunsplit((scheme, netloc, path, parts.query, ""))
    return normalized


def make_event_id(source: str, url: Optional[str], source_id: Optional[str]) -> str:
    """
    Deterministic dedup hash: sha256(source + "|" + normalized_url), falling
    back to sha256(source + "|" + source_id) when there is no URL. Raises if
    neither is available -- a record with no URL and no native id has no
    stable identity and must not be silently assigned a random one.
    """
    norm_url = normalize_url(url)
    if norm_url:
        key = f"{source}|{norm_url}"
    elif source_id:
        key = f"{source}|{source_id}"
    else:
        raise ValueError(
            "Cannot compute a stable dedup id: record has neither a URL nor "
            "a source_id. Fix the connector to always supply one or the "
            "other rather than calling make_event_id with both missing."
        )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass
class RawEvent:
    id: str                      # deterministic dedup hash, see make_event_id()
    source: str                  # "gdelt" | "newsapi" | "rss"
    source_id: Optional[str]     # native id from the source, if it has one
    published_at: str            # ISO 8601 UTC, canonical event/publish time
    ingested_at: str             # ISO 8601 UTC, when this run pulled the record
    title: Optional[str] = None
    text: Optional[str] = None   # body/summary
    url: Optional[str] = None
    raw_metadata: dict = field(default_factory=dict)  # source-specific extras

    def as_dict(self) -> dict:
        return asdict(self)
