// Deterministic 3D layout for CausalityScene3D -- no physics/force-simulation
// dependency (see the plan this shipped under). Per-ticker subgraphs are
// small (typically well under 15 nodes after dedup) and a demo benefits from
// a repeatable layout over force-directed jitter that resettles differently
// on every render.
//
// Nodes are placed on concentric rings by their hop-distance from the focus
// ticker (BFS over the edge list, walked in either direction since edges
// point toward the focus, e.g. "Taiwan" -> "TSM" -> "NVDA"), not by a fixed
// node_type -> ring mapping -- a path can reach the focus via a sector, a
// country directly, or an intermediate stock (e.g. TSM on NVDA's graph), and
// ring-by-hop-distance handles all of those uniformly since propagation
// paths are capped at 2 hops (propagation/graph.py's max_hops).
export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

interface LayoutNode {
  id: string;
}

interface LayoutEdge {
  src: string;
  dst: string;
}

const RING_RADIUS = [0, 4, 7.5];
const Y_JITTER_AMPLITUDE = 0.6;

// Stable per-node jitter seed -- NOT Math.random(), which would resettle the
// scene differently on every re-render/hover and read as jittery rather than
// intentional.
function hashString(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

export function computeCausalGraphLayout(
  nodes: LayoutNode[],
  edges: LayoutEdge[],
  focus: string,
): Record<string, Vec3> {
  const adj = new Map<string, string[]>();
  for (const n of nodes) adj.set(n.id, []);
  for (const e of edges) {
    adj.get(e.src)?.push(e.dst);
    adj.get(e.dst)?.push(e.src);
  }

  const dist = new Map<string, number>([[focus, 0]]);
  const queue: string[] = [focus];
  while (queue.length > 0) {
    const cur = queue.shift()!;
    const d = dist.get(cur)!;
    for (const next of adj.get(cur) ?? []) {
      if (!dist.has(next)) {
        dist.set(next, d + 1);
        queue.push(next);
      }
    }
  }

  const maxRing = RING_RADIUS.length - 1;
  const rings = new Map<number, string[]>();
  for (const n of nodes) {
    // Defensive fallback (should not occur: every node in a causal-graph
    // response is reachable from `focus` by construction) -- park anything
    // unreachable on the outermost ring rather than crashing on undefined.
    const ring = Math.min(dist.get(n.id) ?? maxRing, maxRing);
    if (!rings.has(ring)) rings.set(ring, []);
    rings.get(ring)!.push(n.id);
  }

  const positions: Record<string, Vec3> = {};
  for (const [ring, ids] of rings) {
    const radius = RING_RADIUS[ring];
    ids.forEach((id, i) => {
      const angle = (i / ids.length) * Math.PI * 2;
      const jitter = ring === 0 ? 0 : ((hashString(id) % 100) / 100 - 0.5) * Y_JITTER_AMPLITUDE;
      positions[id] = {
        x: radius * Math.cos(angle),
        y: jitter,
        z: radius * Math.sin(angle),
      };
    });
  }
  return positions;
}
