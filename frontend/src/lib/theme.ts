// lightweight-charts is canvas-rendered and cannot resolve CSS custom
// properties directly -- color options must be literal strings. Reading them
// via getComputedStyle (rather than hand-mirroring the hex values from
// index.css) means this can never drift out of sync with the real palette.
export interface ChartTheme {
  bgPanel: string;
  bgPanelRaised: string;
  border: string;
  text: string;
  textDim: string;
  textHeading: string;
  accent: string;
  accentPurple: string;
  yes: string;
  no: string;
}

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function readChartTheme(): ChartTheme {
  return {
    bgPanel: cssVar("--bg-panel"),
    bgPanelRaised: cssVar("--bg-panel-raised"),
    border: cssVar("--border"),
    text: cssVar("--text"),
    textDim: cssVar("--text-dim"),
    textHeading: cssVar("--text-h"),
    accent: cssVar("--accent"),
    accentPurple: cssVar("--accent-purple"),
    yes: cssVar("--yes"),
    no: cssVar("--no"),
  };
}

// Resolves a "var(--token)" string (as stored in lib/sectorColors.ts and
// lib/eventTypeColors.ts, meant for CSS `color`/`fill` properties that
// understand var() natively) down to a literal computed color. Needed
// specifically for three.js material colors (CausalityScene3D.tsx) -- unlike
// SVG/CSS, three.js's Color constructor cannot resolve var() itself. Values
// that aren't a var() reference are returned unchanged.
export function resolveVar(value: string): string {
  const match = value.match(/^var\((--[\w-]+)\)$/);
  return match ? cssVar(match[1]) : value;
}

// Alpha-blended variants for area/band fills, since CSS custom properties
// here are plain hex and canvas fills need explicit rgba() for translucency.
export function withAlpha(hex: string, alpha: number): string {
  const clean = hex.replace("#", "");
  const r = parseInt(clean.substring(0, 2), 16);
  const g = parseInt(clean.substring(2, 4), 16);
  const b = parseInt(clean.substring(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
