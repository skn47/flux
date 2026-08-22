import { useEffect, useState } from "react";
import { BacktestDrawer } from "./components/BacktestDrawer";
import { CandlestickChart } from "./components/CandlestickChart";
import { CausalityGraphPanel } from "./components/CausalityGraphPanel";
import { EventModal } from "./components/EventModal";
import { HeaderStatus } from "./components/HeaderStatus";
import { OsintFeed } from "./components/OsintFeed";
import { TickerList } from "./components/TickerList";
import { TickerTape } from "./components/TickerTape";
import { WhatIfDrawer } from "./components/WhatIfDrawer";
import { useLiveData } from "./hooks/useLiveData";
import { useQueryParamState } from "./hooks/useQueryParamState";
import { api } from "./services/api";
import { QUICK_RANGES, type QuickRange } from "./lib/chartRanges";
import type { ForecastPoint, PricePoint, Sector, SectorSummary } from "./types";

function rangeStartDate(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 10);
}

function App() {
  const qp = useQueryParamState();
  const live = useLiveData();

  const [sectors, setSectors] = useState<Sector[]>([]);
  const [sectorSummary, setSectorSummary] = useState<SectorSummary | null>(null);
  const [range, setRange] = useState<QuickRange>("6M");
  const [prices, setPrices] = useState<PricePoint[] | null>(null);
  const [forecast, setForecast] = useState<ForecastPoint[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Static/near-static, fetched once.
  useEffect(() => {
    api.sectors().then(setSectors).catch(() => setSectors([]));
    api.sectorSummary().then(setSectorSummary).catch(() => setSectorSummary(null));
  }, []);

  // The Backtests feature is only shown when flux beats baseline Sharpe in
  // EVERY sector -- a deliberate, data-dependent gate (not a bug): this
  // artifact is a periodic snapshot (see api/README.md's "Data freshness"
  // section, scripts/refresh_backtests.sh, weekly), so the button can
  // legitimately appear/disappear across regenerations as the underlying
  // result changes. Starts hidden (false, not null) until the fetch
  // resolves, so there's no flash-then-hide; a fetch failure also resolves
  // to hidden.
  const backtestAvailable =
    sectorSummary != null &&
    sectorSummary.flux_vs_baseline_hit_rate.hits === sectorSummary.flux_vs_baseline_hit_rate.n_sectors;

  // Per-ticker/per-range series. Refetches on ticker/range change, or when
  // useLiveData proves a manual `scripts/refresh_pipeline.sh` rerun actually
  // completed (freshnessVersion) -- never on a blind timer.
  useEffect(() => {
    setError(null);
    const days = QUICK_RANGES.find((r) => r.label === range)?.days ?? 182;
    api
      .prices(qp.ticker, { start: rangeStartDate(days) })
      .then(setPrices)
      .catch((e) => setError(String(e)));
    api.forecast(qp.ticker).then(setForecast).catch(() => setForecast(null));
  }, [qp.ticker, range, live.freshnessVersion]);

  return (
    <div className="app-shell">
      <TickerTape prices={live.latestPrices.size > 0 ? [...live.latestPrices.values()] : []} />

      <header className="app-header">
        <span className="app-wordmark">FLUX TERMINAL</span>
        <div className="app-header-actions">
          <HeaderStatus health={live.health} />
          <button className="header-btn" onClick={() => qp.openPanel("whatif")}>
            What-if
          </button>
          {backtestAvailable && (
            <button className="header-btn" onClick={() => qp.openPanel("backtest")}>
              Backtests
            </button>
          )}
        </div>
      </header>

      <div className="terminal-grid">
        <div className="terminal-left">
          <OsintFeed events={live.osintEvents} onSelectEvent={qp.openEvent} />
        </div>

        <div className="terminal-center panel">
          <div className="panel-header">{qp.ticker} — price</div>
          {error && <p className="error">{error}</p>}
          <CandlestickChart
            points={prices ?? []}
            ticker={qp.ticker}
            forecast={forecast}
            range={range}
            onRangeChange={setRange}
          />

          <div className="panel-header" style={{ marginTop: 14 }}>
            Causality graph — events affecting {qp.ticker}
          </div>
          <CausalityGraphPanel ticker={qp.ticker} onSelectEvent={qp.openEvent} />
        </div>

        <div className="terminal-right">
          <TickerList
            sectors={sectors}
            sparklines={live.sparklines}
            latestPrices={live.latestPrices}
            selectedTicker={qp.ticker}
            onSelectTicker={qp.setTicker}
          />
        </div>
      </div>

      {qp.eventId && <EventModal eventId={qp.eventId} onClose={qp.closeEvent} />}
      {qp.panel === "whatif" && <WhatIfDrawer onClose={qp.closePanel} />}
      {qp.panel === "backtest" && backtestAvailable && (
        <BacktestDrawer onClose={qp.closePanel} sectorSummary={sectorSummary!} />
      )}
    </div>
  );
}

export default App;
