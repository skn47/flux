import { useMemo, useState } from "react";
import { CausalityGraph } from "./CausalityGraph";
import { EventImpactChart } from "./EventImpactChart";
import { api } from "../services/api";
import { EVENT_TYPES } from "../lib/eventTypeColors";
import { SECTOR_COLOR, SECTOR_ORDER } from "../lib/sectorColors";
import type { SimulateResult } from "../types";

const COUNTRIES = ["United States", "Taiwan", "South Korea", "Saudi Arabia", "Russia"];

export function WhatIfSimulator() {
  const [eventType, setEventType] = useState<string>(EVENT_TYPES[2]);
  const [countries, setCountries] = useState<string[]>(["Taiwan"]);
  const [severity, setSeverity] = useState(0.7);
  const [confidence, setConfidence] = useState(0.7);
  const [result, setResult] = useState<SimulateResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const edges = useMemo(
    () =>
      (result?.affected ?? []).map((row) => ({
        id: row.ticker,
        path: row.path,
        weight: row.contribution,
        colorKey: row.sector,
      })),
    [result],
  );

  function toggleCountry(c: string) {
    setCountries((prev) => (prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c]));
  }

  function submit() {
    if (countries.length === 0) {
      setError("Select at least one country.");
      return;
    }
    setLoading(true);
    setError(null);
    api
      .simulate({ eventType, countries, severity, confidence })
      .then(setResult)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }

  return (
    <div className="whatif">
      <label className="whatif-field">
        Event type
        <select value={eventType} onChange={(e) => setEventType(e.target.value)}>
          {EVENT_TYPES.map((t) => (
            <option key={t} value={t}>
              {t.replaceAll("_", " ")}
            </option>
          ))}
        </select>
      </label>
      <div className="whatif-field">
        Countries
        <div className="whatif-countries">
          {COUNTRIES.map((c) => (
            <label key={c} className="whatif-checkbox">
              <input type="checkbox" checked={countries.includes(c)} onChange={() => toggleCountry(c)} />
              {c}
            </label>
          ))}
        </div>
      </div>
      <label className="whatif-field">
        Severity <span className="num">{severity.toFixed(2)}</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={severity}
          onChange={(e) => setSeverity(Number(e.target.value))}
        />
      </label>
      <label className="whatif-field">
        Confidence <span className="num">{confidence.toFixed(2)}</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={confidence}
          onChange={(e) => setConfidence(Number(e.target.value))}
        />
      </label>
      <button className="whatif-submit" onClick={submit} disabled={loading}>
        {loading ? "Recomputing…" : "Simulate"}
      </button>
      {error && <p className="error">{error}</p>}
      {result && (
        <div className="whatif-result">
          <p className="chart-hint">
            Recomputed flux-exposure impact only, via the propagation graph — this
            does not re-run the LSTM model or produce a new price prediction.
          </p>
          <CausalityGraph edges={edges} colorMode="sector" colorMap={SECTOR_COLOR} colorOrder={SECTOR_ORDER} />
          <EventImpactChart affected={result.affected} />
        </div>
      )}
    </div>
  );
}
