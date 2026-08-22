"""
Env-driven config for the Phase H API. Reuses ingestion/env.py's manual .env
loader (existing env vars always win over the .env file) rather than adding a
new settings-management dependency.
"""

from __future__ import annotations

import os

from src.ingestion.env import load_env_file
from src.ingestion.storage import DEFAULT_DB_PATH

load_env_file()

DB_PATH = os.environ.get("API_DB_PATH", str(DEFAULT_DB_PATH))
HOST = os.environ.get("API_HOST", "127.0.0.1")
PORT = int(os.environ.get("API_PORT", "8000"))

# Comma-separated list of allowed frontend origins, e.g.
# "http://localhost:5173,https://example.com". Defaults to both Vite dev
# server origins -- "localhost" and "127.0.0.1" are DIFFERENT origins under
# same-origin policy even though they resolve to the same host (a real bug
# caught by an actual browser test, not just curl, which doesn't enforce
# CORS at all). Deliberately narrow, not "*", since this is a research tool
# with no auth (see api/README.md).
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "API_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]

# How many contributing clusters flux_attribution_daily stores per
# (ticker, date) -- must match flux_engine/timeseries.py's store_daily_attribution
# top_k default. Kept here too so routers/schemas can document the cap without
# importing flux_engine (which pulls in torch-adjacent-weight modules the API
# otherwise avoids).
ATTRIBUTION_TOP_K = 5

# Default page size for GET /api/events (the global recent-events feed).
EVENTS_LIST_DEFAULT_LIMIT = 50
