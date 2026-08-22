# `frontend/` — Phase H research/monitoring dashboard

Vite + React + TypeScript SPA for the Flux Intelligence Platform API
(`api/`, see its own README.md). **Research & monitoring tool — not a
trading signal or investment recommendation.** The disclaimer banner is
rendered once in the root layout (`src/App.tsx`), not per-page, so every
route shows it by construction.

## Run

Requires Node.js (this environment has it via `nvm` — `export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh"`
before any `node`/`npm` command if it's not already on `PATH`).

```
npm install
cp .env.example .env   # only if VITE_API_BASE_URL needs to differ from the default
npm run dev -- --host 127.0.0.1 --port 5173
```

The API (`api/main.py`) must be running separately (default
`http://127.0.0.1:8000`) — this app makes no API calls at build/import time,
only client-side `fetch` calls after mount.

## Verification

```
npm run build         # tsc -b && vite build -- typechecks and production-builds
npm run smoke-test     # real headless-Chromium check (playwright): every
                        # route loads with zero console errors/failed
                        # requests, and the disclaimer banner is present
```

`smoke_test.mjs` is what actually caught two real bugs during development —
worth rerunning after any change to `api/config.py`'s CORS origins or any
route/component:
1. **A CORS misconfiguration** (`http://localhost:5173` and
   `http://127.0.0.1:5173` are different origins under same-origin policy,
   even though they resolve to the same host) — invisible to `curl`, which
   doesn't enforce CORS, only caught by an actual browser fetch.
2. **A `walk_forward_results.json` shape assumption bug**: `sharpe_stats
   .random_rank_mean` has no `median` field (only `baseline`/`flux`/
   `sma_momentum` do), but the table component originally called
   `.median.toFixed()` unconditionally — a real runtime crash, not a
   TypeScript error (the JSON is untyped at the API boundary), only caught
   by loading the actual page in a real browser.

## Structure

- `src/services/api.ts` — the only place that knows the API base URL / fetch
  shape. `src/types.ts` mirrors `api/schemas.py` by hand (kept in sync
  manually — small enough that codegen isn't worth the build complexity).
- `src/components/` — `DisclaimerBanner` (root layout only),
  `FluxScoreChart` (recharts), `EventDrilldownList`, `PredictionPanel`,
  `SectorBacktestTable`, `WalkForwardTable`.
- `src/pages/` — `Overview` (`/`), `TickerDetail` (`/tickers/:ticker`),
  `SectorBacktest` (`/backtest/sectors`), `WalkForward`
  (`/backtest/walk-forward`).

## Known, accepted `npm audit` finding

`react-router` flags a "high" CSRF advisory specific to **RSC (React Server
Components) mode**. This app is a client-only SPA built with `vite build` —
it never enables RSC/SSR — so the advisory's actual attack surface doesn't
apply here. Left unpatched rather than force-downgrading react-router-dom to
an older version for a non-applicable advisory; revisit if this app ever
adds server-side rendering.
