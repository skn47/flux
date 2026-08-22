// Single source of truth for sector display order/color, previously
// duplicated across CascadeGraph.tsx/EventImpactChart.tsx/WhatIfSimulator.tsx.
// Matches config/mvp_scope.py's SECTORS registry.
export const SECTOR_ORDER = ["semiconductors", "financials", "energy", "pharma_biotech"];

export const SECTOR_COLOR: Record<string, string> = {
  semiconductors: "var(--sector-semiconductors)",
  financials: "var(--sector-financials)",
  energy: "var(--sector-energy)",
  pharma_biotech: "var(--sector-pharma_biotech)",
};
