# `api/` — Phase H research/monitoring API

FastAPI backend serving the tracked-stock universe, flux-score history,
event-level "why" drill-down, latest LSTM predictions, and static backtest
findings. **Not a trading signal or investment recommendation** — every
response should be read as research/monitoring output. The walk-forward
validation (`GET /api/backtest/walk-forward`) shows flux beating baseline
Sharpe in only 2 of 4 expanding-window folds — noisy at this sample size,
not a consistent edge (see the endpoint's own scope caveat) — and the
current cross-sector backtest (`GET /api/backtest/sector-summary`) shows the
flux signal beating baseline in 3 of 4 sectors tested.

## Run

```
./.venv/bin/uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive docs at `http://127.0.0.1:8000/docs` (FastAPI's built-in Swagger
UI, auto-generated from `api/schemas.py`).

## Design principles (see the Phase H plan file for full rationale)

- **Every endpoint reads from small, indexed, precomputed tables or cached
  static JSON artifacts** (`flux_scores_daily`, `flux_attribution_daily`,
  `event_catalog`, `latest_predictions`, `backtest/*.json`,
  `lstm/models/run_summary.json`). This process **never** calls
  `src.flux_engine.formula.flux_score()` / `src.flux_engine.query.load_events()`
  (a `classified_events` JOIN `raw_events` scan measured at 14-85s against
  the live 6.4M-row corpus) or runs LSTM inference in a request handler. If
  you're adding an endpoint and find yourself reaching for either of those,
  stop — precompute into a new table via the batch pipeline instead (see
  `src/flux_engine/timeseries.py`'s `store_daily_attribution` /
  `src/backtest/persist_predictions.py` for the existing pattern). The one
  exception is `src.propagation.graph.ExposureGraph.query()` (used by
  `GET /api/events/{event_id}`), which is a pure, zero-I/O, in-memory
  traversal over ~32 nodes/~140 edges — safe to call live, unlike the two
  calls above.
- **Read-only DB access, enforced by SQLite itself**: `api/db.py` opens
  every connection as `file:...?mode=ro`, not just by convention — a write
  attempt raises `sqlite3.OperationalError`, confirmed directly.
- **`data/events.db` runs in WAL journal mode** (migrated as part of Phase H,
  confirmed via `PRAGMA journal_mode`) — a concurrent manual pipeline rerun
  (a writer) no longer blocks API reads.
- **`Infinity`/`NaN` sanitization**: `lstm/models/run_summary.json` contains
  literal `Infinity`/`NaN` tokens (MAPE on exact-zero-return days). Every
  artifact-passthrough response in `routers/reports.py` is sanitized via
  `api/json_safety.py` before leaving the process — raw `json.load()` output
  must never cross the API boundary unsanitized, or a browser's
  `JSON.parse()` will choke on the raw tokens. `null` in a response means
  the underlying metric's denominator was exactly zero.
- **No auth.** Single-deployment, read-only research tool; there is no
  existing auth/rate-limiting convention anywhere in this project to build
  on. If this is ever exposed beyond localhost/a private network, gate it at
  a reverse-proxy layer (e.g. nginx basic auth) — do not add bespoke auth
  to this API.

## Data freshness

**As of 2026-08-19, a cron entry (user `h`) runs `scripts/refresh_pipeline.sh`
once daily at 03:00 local time**, wrapped in `flock -n` against
`/tmp/flux_refresh_pipeline.lock` so a slow run is skipped rather than
overlapped by the next day's trigger (safe either way, since every stage's
own anti-join/`INSERT OR IGNORE` pattern makes a skipped or partial run
idempotent to rerun). Output goes to
`logs/refresh_pipeline/refresh_<date>.log`. This keeps every table below —
including `latest_predictions`/forecasts, not just `event_catalog` — current
to within a day without anyone running anything by hand. `GET /api/health`
reports `latest_flux_score_computed_at` so a caller can see how stale the
data actually is at any moment (e.g. mid-run, or if a cron day was skipped
under `flock`).

Every table this API reads is still, mechanically, populated by a pipeline
*rerun* rather than a live/streaming update — the cron entry just automates
calling that rerun daily. `src/ingestion/scheduler.py` (continuous polling,
default 15 min, matching GDELT's own export cadence — see
`src/ingestion/README.md`) is a separate, unused-by-default tool for
sub-daily raw-ingestion freshness only; it does not touch labeling, price,
flux_engine, features, lstm, or backtest (all under `src/`), so running it
alone does not make `flux_score` or predictions any fresher — use it only if
you want `raw_events` fresher than once a day and are fine with everything
downstream still being on the daily cadence.

For a manual/on-demand refresh (e.g. right before a demo, without waiting for
03:00), run `scripts/refresh_pipeline.sh` from the repo root directly — it runs
the full chain (ingestion → labeling → price → flux_engine → GDELT recheck →
narratives → features → lstm → backtest) in the correct order with one
command. `src.flux_engine.run_timeseries` populates `event_catalog`
automatically as part of the same run (no separate step, unlike
`persist_predictions`). `src.narratives.run_precompute_narratives` (step
7/11) keeps `event_narratives` — the 3D causality graph's per-event AI
explanation, `/api/tickers/{ticker}/causal-graph`'s `narrative` field —
current for whichever events the run_timeseries reruns just promoted into a
ticker's top-5 daily attribution; this was a gap until 2026-08-19 (same shape
as the GDELT-recheck gap above), now closed the same way. If running steps
individually instead, after `src.flux_engine.run_timeseries` and
`src.lstm.run_train`, also run:

```
./.venv/bin/python -m src.backtest.persist_predictions
```

to refresh `latest_predictions` (the `/tickers/{ticker}/prediction`
endpoint's source table). `api/db.py` opens a fresh read-only connection per
request (WAL mode), so the running API server always reflects the latest
refresh on its next request — no restart needed, whether the refresh came
from cron or a manual run.

**Backtests drawer (sector comparison + walk-forward), separate weekly
cadence:** `backtest/sector_metrics.json` and `backtest/walk_forward_
results.json` are NOT part of the nightly `refresh_pipeline.sh` run —
`src.backtest.walk_forward` retrains fresh models from scratch across 4
expanding-window folds, too costly to repeat every night. A separate cron
entry runs `scripts/refresh_backtests.sh` weekly (Sunday 04:30 local, after
the nightly 03:00 job so the two never train models concurrently), also
`flock`-guarded, logging to `logs/refresh_backtests/`. Both artifacts now
carry a `computed_at` timestamp (added 2026-08-20), shown directly in the
Backtests drawer so it's visibly a periodic snapshot rather than implied to
be live. Historical note, since fixed: `api/routers/reports.py`'s
`_load_sanitized` used a bare `@lru_cache` keyed only on file path, so it
would have kept serving the pre-refresh copy forever until the API process
itself restarted — fixed the same day by keying the cache on `(path,
mtime)` instead, so a weekly regeneration is picked up automatically like
every other table here, no restart needed.

## Deployment

Self-hosted (not a managed cloud service): the API runs in Docker on a
machine the operator owns, exposed publicly via **Cloudflare Tunnel** rather
than router port-forwarding (no inbound firewall changes, no home-network
exposure). The frontend deploys separately on Vercel.

```
docker compose up -d
```

builds and starts two containers (`docker-compose.yml`):
- **`api`** — the FastAPI server (`Dockerfile`). Volume-mounts `data/`,
  `lstm/models/`, and `backtest/` straight from the host, so the existing
  crontab-driven pipeline (`scripts/refresh_pipeline.sh` nightly,
  `scripts/refresh_backtests.sh` weekly — unchanged by containerizing the API,
  still running via the host's own `.venv`) keeps writing to the same files
  the container reads. No rebuild or restart needed for a data refresh to
  show up, same WAL/mtime-cache guarantee as running the API directly on the
  host. Deliberately has no `ports:` mapping — not reachable on any host
  network interface, only from inside the compose network.
- **`cloudflared`** — Cloudflare's free Quick Tunnel client, no account or
  domain required. Prints a fresh `https://<random>.trycloudflare.com` URL to
  its logs on every start (`docker compose logs cloudflared`); this only
  changes when the container itself restarts (rare — `restart:
  unless-stopped` means only a host reboot/crash triggers it), but when it
  does, update it in two places: Vercel's `VITE_API_BASE_URL` env var, and
  this API's own `API_CORS_ORIGINS` (set via a repo-root `.env` file, read
  automatically by `docker compose`) so the frontend's origin stays allowed. A
  stable custom hostname is possible instead of Quick Tunnel, but requires a
  domain added to a Cloudflare account — a small ongoing cost not assumed
  here by default.

**CI/CD**: `.github/workflows/deploy.yml` rebuilds and restarts the `api`
container automatically on every push to `main` that touches backend code,
via a **self-hosted GitHub Actions runner installed on the same machine**
(register once via the repo's Settings → Actions → Runners → "New
self-hosted runner", then `svc.sh install && svc.sh start` to persist across
reboots) — same outbound-only trust model as the Cloudflare Tunnel, no
inbound SSH exposure needed. The frontend doesn't need this workflow at all;
Vercel's own GitHub integration rebuilds it natively on every push once the
project is connected there.

## Endpoints

| Method & path | Source |
|---|---|
| `GET /api/health` | `flux_scores_daily` (one `MAX(computed_at)`) |
| `GET /api/sectors`, `/api/tickers` | `config.mvp_scope` (in-memory, no DB) |
| `GET /api/tickers/{ticker}/flux-scores?start=&end=` | `flux_scores_daily` |
| `GET /api/tickers/{ticker}/prices?start=&end=` | `daily_prices` |
| `GET /api/tickers/{ticker}/event-dates?start=&end=` | `flux_attribution_daily` (`rank=1 AND days_since=0`) |
| `GET /api/tickers/{ticker}/events?date=` | `flux_attribution_daily` |
| `GET /api/events?start=&end=&limit=` | `event_catalog` + `flux_attribution_daily` (corridor) |
| `GET /api/events/{event_id}` | `event_catalog` + `flux_attribution_daily` + live `ExposureGraph.query()` |
| `GET /api/tickers/{ticker}/prediction` | `latest_predictions` |
| `GET /api/backtest/sector-summary` | `backtest/sector_metrics.json` |
| `GET /api/backtest/cross-sectional` | `backtest/cross_sectional_metrics.json` |
| `GET /api/backtest/walk-forward` | `backtest/walk_forward_results.json` (+ an injected `scope` caveat field) |
| `GET /api/backtest/run-summary?variant=` | `lstm/models/run_summary.json` |
