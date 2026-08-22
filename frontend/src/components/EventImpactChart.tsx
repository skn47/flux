import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { SECTOR_COLOR, SECTOR_ORDER } from "../lib/sectorColors";

interface ImpactRow {
  ticker: string;
  sector: string;
  contribution: number;
}

interface Props {
  affected: ImpactRow[];
}

export function EventImpactChart({ affected }: Props) {
  if (affected.length === 0) {
    return <p>No tracked tickers were reached by this event.</p>;
  }

  const data = [...affected].sort((a, b) => {
    const sectorDiff = SECTOR_ORDER.indexOf(a.sector) - SECTOR_ORDER.indexOf(b.sector);
    if (sectorDiff !== 0) return sectorDiff;
    return b.contribution - a.contribution;
  });

  const height = Math.max(200, data.length * 28);

  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
          <XAxis type="number" domain={[0, "auto"]} />
          <YAxis type="category" dataKey="ticker" width={56} />
          <Tooltip
            formatter={(value, name) => [
              typeof value === "number" ? value.toFixed(4) : String(value),
              name === "contribution" ? "contribution" : String(name),
            ]}
            labelFormatter={(ticker) => {
              const row = data.find((d) => d.ticker === ticker);
              return row ? `${ticker} — ${row.sector}` : String(ticker);
            }}
          />
          <Bar dataKey="contribution" radius={[0, 4, 4, 0]} isAnimationActive={false}>
            {data.map((row) => (
              <Cell key={row.ticker} fill={SECTOR_COLOR[row.sector] ?? "var(--accent)"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <ul className="chart-legend">
        {SECTOR_ORDER.filter((s) => data.some((d) => d.sector === s)).map((s) => (
          <li key={s}>
            <span className="legend-swatch" style={{ background: SECTOR_COLOR[s] }} />
            {s.replaceAll("_", " ")}
          </li>
        ))}
      </ul>
    </div>
  );
}
