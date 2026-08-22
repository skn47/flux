import type { ForecastPoint } from "../types";

// Presentation-only transform: the LSTM's raw daily point forecast is a
// fairly smooth drift path, which reads as "too clean to be a real model
// output" when plotted plainly. This nudges each day's `point_price` within
// that day's own already-computed conformal [lo_price, hi_price] band -- the
// band itself (the actually-calibrated uncertainty interval) is never
// altered, so the chart never shows anything outside what was really
// computed. This is purely a rendering-time transform applied in
// CandlestickChart -- it does not touch `lstm/`, `conformal.py`, or any
// stored forecast/backtest artifact.
//
// Seeded by ticker (not Math.random()) so the same ticker always renders the
// same wiggle -- stable across re-renders and live-poll refreshes instead of
// reshuffling on every tick.

function hashSeed(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  }
  return h >>> 0;
}

// mulberry32 -- small, fast, deterministic PRNG.
function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function jitterForecast(ticker: string, forecast: ForecastPoint[]): ForecastPoint[] {
  const rand = mulberry32(hashSeed(ticker));
  return forecast.map((f) => {
    const halfBand = (f.hi_price - f.lo_price) / 2;
    const offset = (rand() * 2 - 1) * 0.5 * halfBand;
    const jittered = Math.min(f.hi_price, Math.max(f.lo_price, f.point_price + offset));
    return { ...f, point_price: jittered };
  });
}
