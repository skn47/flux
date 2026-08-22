import { computeDelta } from "../lib/priceDelta";
import type { LatestPrice } from "../types";

interface Props {
  prices: LatestPrice[];
}

export function TickerTape({ prices }: Props) {
  if (prices.length === 0) return null;

  const items = prices.map((p) => {
    const { pct, dir } = computeDelta(p.close, p.prior_close);
    return (
      <span className="tape-item" key={p.ticker}>
        <span className="num tape-ticker">{p.ticker}</span>
        <span className="num">{p.close.toFixed(2)}</span>
        {pct !== null && (
          <span className={`num ${dir}`}>
            {dir === "yes" ? "▲" : "▼"} {Math.abs(pct).toFixed(2)}%
          </span>
        )}
      </span>
    );
  });

  return (
    <div className="ticker-tape">
      <div className="tape-track">
        {items}
        {items}
      </div>
    </div>
  );
}
