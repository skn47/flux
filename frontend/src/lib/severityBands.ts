// Mirrors labeling/schema.py's SEVERITY_BANDS (the project's canonical
// severity_score -> human label mapping) so the frontend doesn't invent its
// own thresholds. Keep in sync with that file if the bands ever change there.
export interface SeverityBand {
  band: number;
  low: number;
  high: number;
  name: string;
  colorVar: string;
}

// Color ramp deliberately avoids --yes (reserved for bullish price direction
// elsewhere in this UI, see eventTypeColors.ts) -- this is a plain
// low-to-high intensity ramp instead: neutral -> amber -> orange -> red.
export const SEVERITY_BANDS: SeverityBand[] = [
  { band: 1, low: 0.0, high: 0.15, name: "negligible", colorVar: "var(--text-dim)" },
  { band: 2, low: 0.15, high: 0.35, name: "minor", colorVar: "var(--border-strong)" },
  { band: 3, low: 0.35, high: 0.55, name: "moderate", colorVar: "var(--cat-amber)" },
  { band: 4, low: 0.55, high: 0.75, name: "major", colorVar: "var(--warn-border)" },
  { band: 5, low: 0.75, high: 1.01, name: "severe", colorVar: "var(--no)" },
];

export function severityBand(score: number): SeverityBand {
  const clipped = Math.max(0, Math.min(1, score));
  return (
    SEVERITY_BANDS.find((b) => clipped >= b.low && clipped < b.high) ??
    SEVERITY_BANDS[SEVERITY_BANDS.length - 1]
  );
}
