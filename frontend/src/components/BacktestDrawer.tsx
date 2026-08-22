import { useEffect, useState } from "react";
import { Drawer } from "./Drawer";
import { SectorBacktestTable } from "./SectorBacktestTable";
import { WalkForwardTable } from "./WalkForwardTable";
import { api } from "../services/api";
import type { SectorSummary, WalkForwardResults } from "../types";

interface Props {
  onClose: () => void;
  // Passed down from App.tsx, which already fetched it once to decide
  // whether to show the Backtests feature at all (gated on flux beating
  // baseline Sharpe in every sector) -- fetching it again here would be a
  // duplicate call for data the caller already has.
  sectorSummary: SectorSummary;
}

type Tab = "sectors" | "walk-forward";

// Ex-pages/SectorBacktest.tsx + pages/WalkForward.tsx, merged into one
// two-tab drawer -- substantive existing work kept (not dropped), just given
// a header-triggered home instead of two routes since neither fits
// naturally into the 3-column layout.
export function BacktestDrawer({ onClose, sectorSummary }: Props) {
  const [tab, setTab] = useState<Tab>("sectors");
  const [walkForward, setWalkForward] = useState<WalkForwardResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.walkForward().then(setWalkForward).catch((e) => setError(String(e)));
  }, []);

  // Both artifacts are static, periodically-regenerated snapshots (see
  // api/README.md's "Data freshness" section -- weekly, not part of the
  // nightly refresh, since a full walk-forward re-run retrains models from
  // scratch across 4 folds). Surfaced here, above the tabs, rather than only
  // inside SectorBacktestTable's prose, so it's visible before reading
  // anything else regardless of which tab is open.
  const computedAt = tab === "sectors" ? sectorSummary.computed_at : walkForward?.computed_at;

  return (
    <Drawer title="Backtests" onClose={onClose}>
      {computedAt && (
        <p className="chart-hint backtest-computed-at">
          Backtest data as of {new Date(computedAt).toLocaleString()} — a periodic snapshot, not
          live.
        </p>
      )}
      <div className="tab-bar">
        <button className={`tab-btn ${tab === "sectors" ? "selected" : ""}`} onClick={() => setTab("sectors")}>
          Sector comparison
        </button>
        <button
          className={`tab-btn ${tab === "walk-forward" ? "selected" : ""}`}
          onClick={() => setTab("walk-forward")}
        >
          Walk-forward findings
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {tab === "sectors" && <SectorBacktestTable summary={sectorSummary} />}
      {tab === "walk-forward" && (
        <div>
          {walkForward && (
            <p className="chart-hint">
              Flux beat baseline Sharpe in{" "}
              {Math.round(walkForward.aggregate.hit_rates.flux_beats_baseline_sharpe * walkForward.aggregate.n_folds)}{" "}
              of {walkForward.aggregate.n_folds} folds. Read the scope caveat below before drawing broader
              conclusions.
            </p>
          )}
          {walkForward ? <WalkForwardTable results={walkForward} /> : <p>Loading…</p>}
        </div>
      )}
    </Drawer>
  );
}
