// Node-link visualization of propagation paths through propagation/graph.py's
// exposure graph. Every `path` is a literal sequence of graph node names --
// e.g. ["Taiwan", "TSM", "NVDA"] -- capped at 2 hops (propagation/graph.py's
// `max_hops=2`), so every path is either a direct 2-node hop (origin ->
// ticker) or a 3-node hop (origin -> intermediate -> ticker). That bound is
// what makes a fixed 3-column layout safe instead of a general force-directed
// graph, matching this codebase's existing preference for hand-rolled SVG
// over a new charting/graph dependency.
//
// Generalized from the original CascadeGraph (one event -> many tickers,
// colored by sector) to a generic edge-list renderer used two ways:
//   - one event -> many tickers, colorMode="sector"   (EventModal, WhatIfDrawer)
//   - many events -> one ticker, colorMode="eventType" (middle-column causality graph)
// The link()/evenlySpacedY() layout helpers are unchanged from the original;
// only the props/color-resolution generalized.
export interface PathEdge {
  id: string;
  path: string[];
  weight: number;
  colorKey: string;
  label?: string;
}

interface Props {
  edges: PathEdge[];
  colorMode: "sector" | "eventType";
  colorMap: Record<string, string>;
  colorOrder?: string[];
  emptyMessage?: string;
}

const W = 760;
const TOP_PAD = 22;
const BOTTOM_PAD = 14;
const ROW_H = 32;
const X0 = 66; // origin column
const X1 = W / 2; // intermediate column
const X2 = W - 66; // destination column

function evenlySpacedY(index: number, count: number, height: number): number {
  const usable = height - TOP_PAD - BOTTOM_PAD;
  const step = usable / count;
  return TOP_PAD + step * (index + 0.5);
}

// Symmetric S-curve between two columns -- standard sankey-link shape.
function link(x1: number, y1: number, x2: number, y2: number): string {
  const cx = x1 + (x2 - x1) / 2;
  return `M ${x1},${y1} C ${cx},${y1} ${cx},${y2} ${x2},${y2}`;
}

export function CausalityGraph({ edges, colorMode, colorMap, colorOrder, emptyMessage }: Props) {
  if (edges.length === 0) {
    return <p>{emptyMessage ?? "No propagation paths on record."}</p>;
  }

  const origins = new Set<string>();
  const mids = new Set<string>();
  const destColor = new Map<string, string>();
  const destWeight = new Map<string, number>();
  const links: { origin: string; mid: string | null; dest: string; weight: number }[] = [];

  for (const e of edges) {
    if (e.path.length === 0) continue;
    const origin = e.path[0];
    const dest = e.path[e.path.length - 1];
    const mid = e.path.length > 2 ? e.path[1] : null;
    origins.add(origin);
    if (mid) mids.add(mid);
    destColor.set(dest, e.colorKey);
    destWeight.set(dest, Math.max(destWeight.get(dest) ?? 0, e.weight));
    links.push({ origin, mid, dest, weight: e.weight });
  }

  const order = colorOrder ?? Object.keys(colorMap);
  const originList = [...origins].sort();
  const midList = [...mids].sort();
  const destList = [...destColor.keys()].sort((a, b) => {
    const diff = order.indexOf(destColor.get(a)!) - order.indexOf(destColor.get(b)!);
    if (diff !== 0) return diff;
    return (destWeight.get(b) ?? 0) - (destWeight.get(a) ?? 0);
  });

  const rows = Math.max(originList.length, midList.length, destList.length, 1);
  const height = TOP_PAD + BOTTOM_PAD + rows * ROW_H;

  const originY = new Map(originList.map((o, i) => [o, evenlySpacedY(i, originList.length, height)]));
  const midY = new Map(midList.map((m, i) => [m, evenlySpacedY(i, midList.length, height)]));
  const destY = new Map(destList.map((d, i) => [d, evenlySpacedY(i, destList.length, height)]));

  const maxWeight = Math.max(...links.map((l) => l.weight), 1e-9);

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${height}`} style={{ width: "100%", height: "auto", display: "block" }}>
        {links.map((l, i) => {
          const y1 = originY.get(l.origin)!;
          const y2 = destY.get(l.dest)!;
          const strokeWidth = Math.max(1, (l.weight / maxWeight) * 5);
          const color = colorMap[destColor.get(l.dest) ?? ""] ?? "var(--text-dim)";
          if (l.mid) {
            const yMid = midY.get(l.mid)!;
            return (
              <g key={i} opacity={0.55}>
                <path d={link(X0, y1, X1, yMid)} fill="none" stroke={color} strokeWidth={strokeWidth} />
                <path d={link(X1, yMid, X2, y2)} fill="none" stroke={color} strokeWidth={strokeWidth} />
              </g>
            );
          }
          return (
            <path key={i} d={link(X0, y1, X2, y2)} fill="none" stroke={color} strokeWidth={strokeWidth} opacity={0.55} />
          );
        })}

        {originList.map((o) => (
          <g key={`o-${o}`}>
            <circle cx={X0} cy={originY.get(o)} r={6} fill="var(--text-dim)" stroke="var(--bg-void)" strokeWidth={2} />
            <text x={X0 + 12} y={originY.get(o)! + 4} fontSize={11} fill="var(--text)">
              {o}
            </text>
          </g>
        ))}

        {midList.map((m) => (
          <g key={`m-${m}`}>
            <circle cx={X1} cy={midY.get(m)} r={5} fill="var(--border-strong)" stroke="var(--bg-void)" strokeWidth={2} />
            <text x={X1} y={midY.get(m)! - 11} fontSize={10} fill="var(--text-dim)" textAnchor="middle">
              {m.replaceAll("_", " ")}
            </text>
          </g>
        ))}

        {destList.map((d) => {
          const color = colorMap[destColor.get(d) ?? ""] ?? "var(--accent)";
          return (
            <g key={`d-${d}`}>
              <circle cx={X2} cy={destY.get(d)} r={7} fill={color} stroke="var(--bg-void)" strokeWidth={2} />
              <text x={X2 - 13} y={destY.get(d)! + 4} fontSize={11} fontWeight={600} fill="var(--text-h)" textAnchor="end">
                {colorMode === "sector" ? d : d.length > 14 ? `${d.slice(0, 13)}…` : d}
              </text>
            </g>
          );
        })}
      </svg>
      <ul className="chart-legend">
        {order
          .filter((key) => [...destColor.values()].includes(key))
          .map((key) => (
            <li key={key}>
              <span className="legend-swatch" style={{ background: colorMap[key] }} />
              {key.replaceAll("_", " ")}
            </li>
          ))}
      </ul>
      <p className="chart-hint">
        {colorMode === "sector"
          ? "Propagation path from the event's origin (left) through the exposure graph to each affected ticker (right)."
          : "Propagation path from each contributing event's origin (left) through the exposure graph to this ticker (right)."}{" "}
        Line thickness scales with contribution.
        {midList.length === 0 ? " Every path here was a single hop." : ""}
      </p>
    </div>
  );
}
