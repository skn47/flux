import { computeDelta } from "../lib/priceDelta";
import type { LatestPrice, Sector, Sparkline } from "../types";

interface Props {
  sectors: Sector[];
  sparklines: Map<string, Sparkline>;
  latestPrices: Map<string, LatestPrice>;
  selectedTicker?: string;
  onSelectTicker: (ticker: string) => void;
}

const SPARK_W = 64;
const SPARK_H = 22;

function sparklinePath(values: number[]): string {
  if (values.length < 2) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const stepX = SPARK_W / (values.length - 1);
  return values
    .map((v, i) => {
      const x = i * stepX;
      const y = SPARK_H - ((v - min) / range) * SPARK_H;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

// Renamed from SectorMatrix.tsx -- `.matrix-grid` was already a scrollable
// flex column (not the unused `.sector-grid`), so this is "extend the
// right-column ticker list with a live price," not "replace a grid."
// Sectors/sparklines/latestPrices now come from useLiveData (App.tsx) rather
// than each being fetched independently here.
export function TickerList({ sectors, sparklines, latestPrices, selectedTicker, onSelectTicker }: Props) {
  if (sectors.length === 0) return <p>Loading…</p>;

  return (
    <div className="matrix-grid">
      {sectors.map((s) => (
        <div key={s.sector} className="matrix-sector">
          <div className="panel-header">{s.sector.replaceAll("_", " ")}</div>
          <div className="matrix-tickers">
            {s.tickers.map((t) => {
              const spark = sparklines.get(t);
              const values = spark?.points.map((p) => p.adj_close) ?? [];
              const up = values.length >= 2 && values[values.length - 1] >= values[0];
              const price = latestPrices.get(t);
              const { pct, dir } = price ? computeDelta(price.close, price.prior_close) : { pct: null, dir: "" as const };
              return (
                <button
                  key={t}
                  className={`matrix-block ${t === selectedTicker ? "selected" : ""}`}
                  style={{ borderLeftColor: `var(--sector-${s.sector})` }}
                  onClick={() => onSelectTicker(t)}
                >
                  <span className="num matrix-ticker">{t}</span>
                  {values.length >= 2 && (
                    <svg width={SPARK_W} height={SPARK_H} className="matrix-spark">
                      <path
                        d={sparklinePath(values)}
                        fill="none"
                        stroke={up ? "var(--yes)" : "var(--no)"}
                        strokeWidth={1.5}
                      />
                    </svg>
                  )}
                  <span className="matrix-price-block">
                    {price && <span className="num matrix-price">{price.close.toFixed(2)}</span>}
                    {pct !== null && (
                      <span className={`num matrix-delta ${dir}`}>
                        {dir === "yes" ? "▲" : "▼"} {Math.abs(pct).toFixed(2)}%
                      </span>
                    )}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
