// Single source of truth for the 8 known event types (moved out of
// WhatIfSimulator.tsx, which used to own this list alone) and their display
// colors. Deliberately does NOT reuse `--yes`/`--no` (they carry a strong
// up/down directional meaning elsewhere in this UI -- coloring, say, a
// "natural_disaster" event green would read as "bullish") -- two new
// categorical tokens (`--cat-amber`, `--cat-rose`) were added to index.css
// instead, alongside the 4 sector hues and the 2 accent hues.
export const EVENT_TYPES = [
  "supply_chain_fab_disruption",
  "natural_disaster",
  "geopolitical_military_tension",
  "trade_export_control",
  "regulatory_approval_decision",
  "regulatory_subsidy_action",
  "macroeconomic_conditions",
  "corporate_strategic",
] as const;

export const EVENT_TYPE_COLOR: Record<string, string> = {
  supply_chain_fab_disruption: "var(--sector-semiconductors)",
  natural_disaster: "var(--sector-energy)",
  geopolitical_military_tension: "var(--cat-rose)",
  trade_export_control: "var(--sector-financials)",
  regulatory_approval_decision: "var(--sector-pharma_biotech)",
  regulatory_subsidy_action: "var(--cat-amber)",
  macroeconomic_conditions: "var(--accent)",
  corporate_strategic: "var(--accent-purple)",
};
