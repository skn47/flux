import { useCallback, useEffect, useState } from "react";

export type PanelKind = "whatif" | "backtest";

interface QueryParamState {
  ticker: string;
  eventId: string | undefined;
  panel: PanelKind | undefined;
}

const DEFAULT_TICKER = "NVDA";

function readParams(): QueryParamState {
  const qs = new URLSearchParams(window.location.search);
  const panel = qs.get("panel");
  return {
    ticker: qs.get("ticker") ?? DEFAULT_TICKER,
    eventId: qs.get("event") ?? undefined,
    panel: panel === "whatif" || panel === "backtest" ? panel : undefined,
  };
}

function writeParams(next: QueryParamState): void {
  const qs = new URLSearchParams();
  if (next.ticker && next.ticker !== DEFAULT_TICKER) qs.set("ticker", next.ticker);
  if (next.eventId) qs.set("event", next.eventId);
  if (next.panel) qs.set("panel", next.panel);
  const search = qs.toString();
  const url = `${window.location.pathname}${search ? `?${search}` : ""}`;
  window.history.pushState(null, "", url);
}

// Replaces react-router-dom's route params now that there is exactly one
// page: `?ticker=&event=&panel=` are the single source of truth for what's
// selected/open, kept in sync with the URL via history.pushState (no reload)
// so deep links / back-button behavior still work without a router
// dependency.
export function useQueryParamState() {
  const [state, setState] = useState<QueryParamState>(readParams);

  useEffect(() => {
    const onPopState = () => setState(readParams());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const update = useCallback((patch: Partial<QueryParamState>) => {
    setState((prev) => {
      const next = { ...prev, ...patch };
      writeParams(next);
      return next;
    });
  }, []);

  const setTicker = useCallback((t: string) => update({ ticker: t }), [update]);
  const openEvent = useCallback((id: string) => update({ eventId: id }), [update]);
  const closeEvent = useCallback(() => update({ eventId: undefined }), [update]);
  const openPanel = useCallback((p: PanelKind) => update({ panel: p }), [update]);
  const closePanel = useCallback(() => update({ panel: undefined }), [update]);

  return {
    ticker: state.ticker,
    eventId: state.eventId,
    panel: state.panel,
    setTicker,
    openEvent,
    closeEvent,
    openPanel,
    closePanel,
  };
}
