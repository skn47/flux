import { Component, lazy, Suspense, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { CausalityGraph } from "./CausalityGraph";
import { api } from "../services/api";
import { EVENT_TYPE_COLOR, EVENT_TYPES } from "../lib/eventTypeColors";
import type { CausalGraphOut } from "../types";

const CausalityScene3D = lazy(() =>
  import("./CausalityScene3D").then((m) => ({ default: m.CausalityScene3D })),
);

interface Props {
  ticker: string;
  onSelectEvent: (eventId: string) => void;
}

// Fallback path shared by both the Suspense loading state (the 3D chunk is
// still downloading -- see vite.config.ts's manualChunks) and the error
// boundary (WebGL context creation failed, or the scene threw). Reusing the
// pre-existing 2D CausalityGraph here means: no new skeleton/spinner needed,
// a broken 3D scene degrades to the exact chart that existed before this
// feature rather than crashing, and there's a genuine "2D snaps into 3D"
// first-paint moment once the chunk finishes loading.
function fallbackEdges(graph: CausalGraphOut) {
  return graph.events.map((e) => ({
    id: `${e.event_id}-${e.rank}`,
    path: e.path,
    weight: e.contribution,
    colorKey: e.event_type,
    label: e.event_type,
  }));
}

function Fallback({ graph }: { graph: CausalGraphOut }) {
  return (
    <CausalityGraph
      edges={fallbackEdges(graph)}
      colorMode="eventType"
      colorMap={EVENT_TYPE_COLOR}
      colorOrder={[...EVENT_TYPES]}
      emptyMessage="No contributing events on record for the latest scored date."
    />
  );
}

class CausalitySceneErrorBoundary extends Component<
  { graph: CausalGraphOut; children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: unknown) {
    console.error("CausalityScene3D failed, falling back to the 2D graph:", error);
  }

  render() {
    if (this.state.failed) return <Fallback graph={this.props.graph} />;
    return this.props.children;
  }
}

export function CausalityGraphPanel({ ticker, onSelectEvent }: Props) {
  const [graph, setGraph] = useState<CausalGraphOut | null>(null);

  useEffect(() => {
    setGraph(null);
    api.causalGraph(ticker).then(setGraph).catch(() => setGraph(null));
  }, [ticker]);

  if (!graph) return <p>Loading…</p>;

  return (
    <CausalitySceneErrorBoundary graph={graph}>
      <Suspense fallback={<Fallback graph={graph} />}>
        <CausalityScene3D graph={graph} onSelectEvent={onSelectEvent} />
      </Suspense>
    </CausalitySceneErrorBoundary>
  );
}
