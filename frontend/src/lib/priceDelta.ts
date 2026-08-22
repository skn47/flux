export interface PriceDelta {
  pct: number | null;
  dir: "yes" | "no" | "";
}

// Shared by TickerTape and TickerList so both compute the same last-close vs.
// prior-close delta/direction instead of duplicating the math.
export function computeDelta(close: number, priorClose: number | null): PriceDelta {
  if (priorClose === null) return { pct: null, dir: "" };
  const delta = close - priorClose;
  const pct = priorClose ? (delta / priorClose) * 100 : null;
  const dir = delta >= 0 ? "yes" : "no";
  return { pct, dir };
}
