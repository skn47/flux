import type {
  AttributionEvent,
  CausalGraphOut,
  EventDetail,
  EventSummary,
  ForecastPoint,
  Health,
  LatestPrice,
  PricePoint,
  Sector,
  SectorSummary,
  SimulateResult,
  Sparkline,
  Ticker,
  WalkForwardResults,
} from "../types";

const BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`${path} -> HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => get<Health>("/api/health"),
  sectors: () => get<Sector[]>("/api/sectors"),
  tickers: () => get<Ticker[]>("/api/tickers"),
  prices: (ticker: string, opts?: { start?: string; end?: string }) => {
    const qs = new URLSearchParams();
    if (opts?.start) qs.set("start", opts.start);
    if (opts?.end) qs.set("end", opts.end);
    const suffix = qs.toString();
    return get<PricePoint[]>(`/api/tickers/${ticker}/prices${suffix ? `?${suffix}` : ""}`);
  },
  events: (ticker: string, date?: string) =>
    get<AttributionEvent[]>(
      `/api/tickers/${ticker}/events${date ? `?date=${date}` : ""}`,
    ),
  causalGraph: (ticker: string, date?: string) =>
    get<CausalGraphOut>(`/api/tickers/${ticker}/causal-graph${date ? `?date=${date}` : ""}`),
  forecast: (ticker: string) =>
    get<ForecastPoint[]>(`/api/tickers/${ticker}/forecast`),
  eventDetail: (eventId: string) => get<EventDetail>(`/api/events/${eventId}`),
  eventsList: (limit?: number) =>
    get<EventSummary[]>(`/api/events${limit ? `?limit=${limit}` : ""}`),
  latestPrices: () => get<LatestPrice[]>("/api/tickers/latest-prices"),
  sparklines: (days?: number) =>
    get<Sparkline[]>(`/api/tickers/sparklines${days ? `?days=${days}` : ""}`),
  simulate: (params: {
    eventType: string;
    countries: string[];
    severity?: number;
    confidence?: number;
  }) => {
    const qs = new URLSearchParams();
    qs.set("event_type", params.eventType);
    for (const c of params.countries) qs.append("countries", c);
    if (params.severity !== undefined) qs.set("severity", String(params.severity));
    if (params.confidence !== undefined) qs.set("confidence", String(params.confidence));
    return get<SimulateResult>(`/api/simulate?${qs.toString()}`);
  },
  sectorSummary: () => get<SectorSummary>("/api/backtest/sector-summary"),
  walkForward: () => get<WalkForwardResults>("/api/backtest/walk-forward"),
};
