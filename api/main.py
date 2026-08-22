"""
Phase H API entrypoint. Run with:
    ./.venv/bin/uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
(or ./.venv/bin/python -m api.main)

Research/monitoring tool -- see api/README.md and the frontend's persistent
disclaimer banner. This API deliberately never calls
flux_engine.formula.flux_score()/load_events() or runs LSTM inference in a
request handler; every endpoint reads from small, indexed, precomputed
tables or cached static JSON artifacts. See routers/*.py docstrings for the
per-endpoint data source.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from api.config import CORS_ORIGINS, HOST, PORT
from api.routers import events, forecast, meta, predictions, reports, flux, simulate

app = FastAPI(
    title="Flux Intelligence Platform API",
    description=(
        "Research & monitoring API for tracked-stock flux scores, event "
        "drill-down, LSTM predictions, and backtest findings. Not a trading "
        "signal or investment recommendation -- see /api/health and each "
        "endpoint's response for evidentiary caveats."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)
# Every response here is JSON from small precomputed tables, except /prices
# (still hundreds of KB even after its default-window fix below) -- gzip cuts
# that by ~10x and costs nothing on these tiny payload sizes. min_size=500
# skips compressing already-small responses (health checks, etc.) where the
# gzip framing overhead isn't worth it.
app.add_middleware(GZipMiddleware, minimum_size=500)

app.include_router(meta.router)
app.include_router(flux.router)
app.include_router(events.router)
app.include_router(predictions.router)
app.include_router(forecast.router)
app.include_router(reports.router)
app.include_router(simulate.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=HOST, port=PORT, reload=False)
