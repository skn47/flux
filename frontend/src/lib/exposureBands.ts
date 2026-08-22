// Weak/Moderate/Strong tiers for per-ticker impact and contribution, so
// EventModal can show a plain-language label instead of a raw decimal.
//
// Thresholds are tercile cuts (33rd/67th percentile) measured directly
// against the live corpus (2026-08-16), not guessed:
//   - imp: queried flux_attribution_daily.imp (n=87,164) -> cuts at 0.25 / 0.48
//   - contribution: the live /api/events/{id} endpoint computes this as
//     k_value * imp (no tau -- see api/routers/events.py, this differs from
//     the tau-inclusive `contribution` stored in flux_attribution_daily and
//     used by /api/tickers/{ticker}/events). Queried k_value * imp directly
//     to match the live formula -- cuts at 0.058 / 0.084.
// These are a first calibration off the current corpus, not permanent
// constants -- same tunable-with-rationale posture as
// labeling/ollama_labeler.py's MIN_CONFIRM_CONFIDENCE. Revisit if the corpus
// composition shifts materially (e.g. after the geopolitical_military_tension
// gate redesign changes the mix of events reaching this endpoint).

export interface Tier {
  label: "Weak" | "Moderate" | "Strong";
  colorVar: string;
}

const WEAK: Tier = { label: "Weak", colorVar: "var(--text-dim)" };
const MODERATE: Tier = { label: "Moderate", colorVar: "var(--cat-amber)" };
const STRONG: Tier = { label: "Strong", colorVar: "var(--warn-border)" };

const IMPACT_LOW = 0.25;
const IMPACT_HIGH = 0.48;

const CONTRIBUTION_LOW = 0.058;
const CONTRIBUTION_HIGH = 0.084;

export function impactTier(imp: number): Tier {
  if (imp < IMPACT_LOW) return WEAK;
  if (imp < IMPACT_HIGH) return MODERATE;
  return STRONG;
}

export function contributionTier(contribution: number): Tier {
  if (contribution < CONTRIBUTION_LOW) return WEAK;
  if (contribution < CONTRIBUTION_HIGH) return MODERATE;
  return STRONG;
}
