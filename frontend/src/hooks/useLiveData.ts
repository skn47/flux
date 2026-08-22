// The one place in the app that polls. `ingestion/scheduler.py` (a separate
// piece of work) can keep `raw_events` fresh on its own schedule, but it does
// NOT refresh `flux_score`/predictions/forecasts -- those still require a
// manual `scripts/refresh_pipeline.sh` rerun. Polling here only avoids
// showing a stale page load if that rerun has, in fact, already happened
// since the page opened; it never creates new underlying data on its own.
// Concretely: ticker tape / sparklines / the OSINT feed poll on a flat
// interval (cheap, DB-indexed reads), while the selected ticker's
// flux-scores/predictions/forecast are only refetched when `health`'s
// `latest_flux_score_computed_at` actually changes, or when the ticker
// itself changes -- never on a blind timer.
import { useCallback, useRef, useState } from "react";
import { usePolling } from "./usePolling";
import { api } from "../services/api";
import type { EventSummary, Health, LatestPrice, Sparkline } from "../types";

const PRICE_FEED_INTERVAL_MS = 45_000;
const HEALTH_INTERVAL_MS = 120_000;

export interface LiveData {
  latestPrices: Map<string, LatestPrice>;
  sparklines: Map<string, Sparkline>;
  osintEvents: EventSummary[];
  health: Health | null;
  freshnessVersion: number;
}

export function useLiveData(): LiveData {
  const [latestPrices, setLatestPrices] = useState<Map<string, LatestPrice>>(new Map());
  const [sparklines, setSparklines] = useState<Map<string, Sparkline>>(new Map());
  const [osintEvents, setOsintEvents] = useState<EventSummary[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [freshnessVersion, setFreshnessVersion] = useState(0);
  const lastComputedAt = useRef<string | null>(null);

  const fetchFeeds = useCallback(() => {
    Promise.all([api.latestPrices(), api.sparklines(30), api.eventsList(50)])
      .then(([prices, sparks, events]) => {
        setLatestPrices(new Map(prices.map((p) => [p.ticker, p])));
        setSparklines(new Map(sparks.map((s) => [s.ticker, s])));
        setOsintEvents(events);
      })
      .catch(() => {
        // A failed poll just leaves the previous snapshot on screen -- no
        // error state needed for a background refresh.
      });
  }, []);

  const fetchHealth = useCallback(() => {
    api
      .health()
      .then((h) => {
        setHealth(h);
        if (h.latest_flux_score_computed_at !== lastComputedAt.current) {
          lastComputedAt.current = h.latest_flux_score_computed_at;
          setFreshnessVersion((v) => v + 1);
        }
      })
      .catch(() => {});
  }, []);

  usePolling(fetchFeeds, PRICE_FEED_INTERVAL_MS);
  usePolling(fetchHealth, HEALTH_INTERVAL_MS);

  return { latestPrices, sparklines, osintEvents, health, freshnessVersion };
}
