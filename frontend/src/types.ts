// Mirrors api/schemas.py exactly -- keep in sync by hand (small enough that a
// codegen step isn't worth the added build complexity for this MVP).

export interface Ticker {
  ticker: string;
  sector: string;
  company_names: string[];
}

export interface Sector {
  sector: string;
  tickers: string[];
}

export interface FluxScorePoint {
  date: string;
  flux_score: number;
  direction: number | null;
  direction_coverage: number;
  n_clusters: number;
}

export interface AttributionEvent {
  rank: number;
  event_id: string;
  label_source: string;
  event_type: string;
  corridor: string[];
  path: string[];
  cluster_date: string;
  contribution: number;
  imp: number;
  tau: number;
  days_since: number;
  k_value: number;
}

export interface CausalGraphNode {
  id: string;
  node_type: "country" | "sector" | "stock";
  is_focus: boolean;
}

export interface CausalGraphEdge {
  src: string;
  dst: string;
  weight: number;
  channel: string | null;
  note: string;
}

export interface CausalGraphEvent {
  event_id: string;
  rank: number;
  event_type: string;
  label_source: string;
  title: string | null;
  narrative: string | null;
  path: string[];
  contribution: number;
  cluster_date: string;
}

export interface CausalGraphOut {
  ticker: string;
  date: string;
  nodes: CausalGraphNode[];
  edges: CausalGraphEdge[];
  events: CausalGraphEvent[];
}

export interface PricePoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  adj_close: number;
  volume: number;
}

export interface EventDateMarker {
  date: string;
  event_id: string;
  event_type: string;
  label_source: string;
  corridor: string[];
}

export interface EventSummary {
  event_id: string;
  event_type: string;
  label_source: string;
  severity_score: number;
  published_at: string;
  title: string | null;
  corridor: string[];
}

export interface EventTickerImpact {
  ticker: string;
  sector: string;
  imp: number;
  path: string[];
  contribution: number;
  ever_top5: boolean;
}

export interface EventDetail {
  event_id: string;
  event_type: string;
  label_source: string;
  severity_score: number;
  confidence: number;
  polarity: number | null;
  countries: string[];
  corridor: string[];
  published_at: string;
  url: string | null;
  title: string | null;
  k_value: number;
  affected: EventTickerImpact[];
}

export interface Prediction {
  ticker: string;
  variant: "baseline" | "flux";
  label_date: string;
  prior_date: string;
  actual_return_L: number | null;
  predicted_return_L: number;
  computed_at: string;
}

export interface ForecastPoint {
  ticker: string;
  variant: string;
  as_of_date: string;
  horizon_step: number;
  target_date: string;
  point_price: number;
  lo_price: number;
  hi_price: number;
  computed_at: string;
}

export interface LatestPrice {
  ticker: string;
  date: string;
  close: number;
  adj_close: number;
  prior_close: number | null;
}

export interface SparklinePoint {
  date: string;
  adj_close: number;
}

export interface Sparkline {
  ticker: string;
  points: SparklinePoint[];
}

export interface SimulateTickerImpact {
  ticker: string;
  sector: string;
  imp: number;
  path: string[];
  contribution: number;
}

export interface SimulateResult {
  event_type: string;
  countries: string[];
  severity: number;
  confidence: number;
  affected: SimulateTickerImpact[];
}

export interface Health {
  status: string;
  db_path_ok: boolean;
  latest_flux_score_computed_at: string | null;
}

// lstm/models/run_summary.json passthrough shape (only the fields the UI uses)
export interface RunSummary {
  [variant: string]: {
    metrics?: {
      test?: {
        per_ticker?: Record<string, { n: number; mae: number; rmse: number; mape: number }>;
      };
    };
  };
}

// backtest/sector_metrics.json passthrough shape (only the fields the UI uses)
export interface SectorSummary {
  computed_at?: string;
  dataset_version_note?: string;
  test_window?: { start: string; end: string };
  flux_vs_baseline_hit_rate: {
    per_sector: {
      sector: string;
      baseline_sharpe: number;
      flux_sharpe: number;
      flux_beats_baseline: boolean;
    }[];
    hits: number;
    n_sectors: number;
  };
}

// backtest/walk_forward_results.json passthrough shape (only the fields the UI uses)
export interface WalkForwardResults {
  computed_at?: string;
  scope: string;
  folds: {
    fold_id: number;
    train: [string, string];
    val: [string, string];
    test: [string, string];
  }[];
  aggregate: {
    hit_rates: Record<string, number>;
    sharpe_stats: Record<
      string,
      { per_fold: number[]; mean: number; median?: number; std?: number }
    >;
    n_folds: number;
  };
}
