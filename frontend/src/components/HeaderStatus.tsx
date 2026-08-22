import { useEffect, useState } from "react";
import type { Health } from "../types";

interface Props {
  health: Health | null;
}

// Nightly refresh cron runs every 24h (scripts/refresh_pipeline.sh, see
// api/README.md's "Data freshness" section) -- 30h gives slack for a run
// finishing a few hours late under flock, without flagging "stale" on an
// ordinary day.
const STALE_THRESHOLD_MS = 30 * 60 * 60 * 1000;

function formatTime(d: Date): string {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

// Surfaces the /api/health data useLiveData already polls (every 120s) but
// that nothing in the UI previously showed -- a live/stale indicator instead
// of a purely decorative header filler. The ticking clock below is the one
// purely cosmetic piece; it runs off its own 1s interval, no network calls.
export function HeaderStatus({ health }: Props) {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const computedAt = health?.latest_flux_score_computed_at ?? null;
  const hasHealth = health != null && computedAt != null;
  const isLive =
    hasHealth &&
    health!.db_path_ok &&
    now.getTime() - new Date(computedAt!).getTime() < STALE_THRESHOLD_MS;

  return (
    <div className="header-status">
      {hasHealth && (
        <>
          <span className={`status-dot ${isLive ? "status-dot-live" : "status-dot-stale"}`} />
          <span className={`status-label ${isLive ? "status-label-live" : "status-label-stale"}`}>
            {isLive ? "LIVE" : "STALE"}
          </span>
          <span className="status-detail num">Data as of {formatTime(new Date(computedAt!))}</span>
          <span className="header-divider" />
        </>
      )}
      <span className="header-clock num">{formatTime(now)}</span>
    </div>
  );
}
