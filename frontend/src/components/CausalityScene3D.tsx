import { useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html, Instances, Instance, type PositionMesh } from "@react-three/drei";
import * as THREE from "three";
import { computeCausalGraphLayout, type Vec3 } from "../lib/causalGraphLayout";
import { resolveVar } from "../lib/theme";
import { SECTOR_COLOR } from "../lib/sectorColors";
import { EVENT_TYPE_COLOR, EVENT_TYPES } from "../lib/eventTypeColors";
import type { CausalGraphOut, CausalGraphEvent } from "../types";

interface Props {
  graph: CausalGraphOut;
  onSelectEvent: (eventId: string) => void;
}

const V = (v: Vec3) => new THREE.Vector3(v.x, v.y, v.z);

function nodeColor(nodeId: string, nodeType: string, isFocus: boolean): string {
  if (isFocus) return resolveVar("var(--accent)");
  if (nodeType === "sector") return resolveVar(SECTOR_COLOR[nodeId] ?? "var(--text-dim)");
  return resolveVar("var(--text-dim)");
}

// The highest-contribution event whose path walks this exact src->dst step
// -- used both to color the edge (mirrors the old 2D CausalityGraph's
// colorMode="eventType") and as the event a click on this edge opens.
function primaryEventForEdge(events: CausalGraphEvent[], src: string, dst: string): CausalGraphEvent | null {
  let best: CausalGraphEvent | null = null;
  for (const e of events) {
    for (let i = 0; i < e.path.length - 1; i++) {
      if (e.path[i] === src && e.path[i + 1] === dst) {
        if (!best || e.contribution > best.contribution) best = e;
      }
    }
  }
  return best;
}

function CameraRig() {
  const { camera } = useThree();
  const start = useRef(performance.now());
  const from = useMemo(() => new THREE.Vector3(0, 16, 24), []);
  const to = useMemo(() => new THREE.Vector3(0, 5, 11), []);
  const done = useRef(false);

  useFrame(() => {
    if (done.current) return;
    const t = Math.min(1, (performance.now() - start.current) / 1200);
    const eased = 1 - Math.pow(1 - t, 3); // cubic ease-out
    camera.position.lerpVectors(from, to, eased);
    camera.lookAt(0, 0, 0);
    if (t >= 1) done.current = true;
  });
  return null;
}

function EdgeParticles({ curve, color, weight }: { curve: THREE.QuadraticBezierCurve3; color: string; weight: number }) {
  const count = 3;
  const refs = useRef<(PositionMesh | null)[]>([]);
  const speed = 0.15 + weight * 0.35;

  useFrame((state) => {
    const elapsed = state.clock.getElapsedTime();
    for (let i = 0; i < count; i++) {
      const obj = refs.current[i];
      if (!obj) continue;
      const t = (elapsed * speed + i / count) % 1;
      const p = curve.getPointAt(t);
      obj.position.copy(p);
    }
  });

  return (
    <Instances limit={count}>
      <sphereGeometry args={[0.06 + weight * 0.05, 8, 8]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.5} toneMapped={false} />
      {Array.from({ length: count }, (_, i) => (
        <Instance
          key={i}
          ref={(el: PositionMesh | null) => {
            refs.current[i] = el;
          }}
        />
      ))}
    </Instances>
  );
}

function Edge({
  src,
  dst,
  weight,
  positions,
  event,
  hovered,
  onHover,
  onClick,
}: {
  src: string;
  dst: string;
  weight: number;
  positions: Record<string, Vec3>;
  event: CausalGraphEvent | null;
  hovered: boolean;
  onHover: (key: string | null) => void;
  onClick: () => void;
}) {
  const color = resolveVar(event ? EVENT_TYPE_COLOR[event.event_type] ?? "var(--text-dim)" : "var(--text-dim)");
  const curve = useMemo(() => {
    const a = V(positions[src]);
    const b = V(positions[dst]);
    const mid = a.clone().add(b).multiplyScalar(0.5);
    mid.y += 0.8; // gentle arc, matches the old sankey's curved-link feel
    return new THREE.QuadraticBezierCurve3(a, mid, b);
  }, [positions, src, dst]);

  const tubeRadius = 0.03 + weight * 0.06;
  const geometry = useMemo(() => new THREE.TubeGeometry(curve, 24, tubeRadius, 8, false), [curve, tubeRadius]);

  return (
    <group>
      <mesh
        geometry={geometry}
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover(`${src}->${dst}`);
        }}
        onPointerOut={() => onHover(null)}
        onClick={(e) => {
          e.stopPropagation();
          onClick();
        }}
      >
        <meshStandardMaterial
          color={color}
          transparent
          opacity={hovered ? 0.95 : 0.55}
          emissive={color}
          emissiveIntensity={hovered ? 0.6 : 0.15}
        />
      </mesh>
      <EdgeParticles curve={curve} color={color} weight={weight} />
      {hovered && (
        <Html position={curve.getPointAt(0.5)} center distanceFactor={12}>
          <div className="causal-hover-label">
            {event ? event.event_type.replaceAll("_", " ") : `${src} → ${dst}`}
          </div>
        </Html>
      )}
    </group>
  );
}

function Node({
  id,
  nodeType,
  isFocus,
  position,
  maxContribution,
  hovered,
  onHover,
}: {
  id: string;
  nodeType: string;
  isFocus: boolean;
  position: Vec3;
  maxContribution: number;
  hovered: boolean;
  onHover: (id: string | null) => void;
}) {
  const color = nodeColor(id, nodeType, isFocus);
  const radius = isFocus ? 0.55 : 0.28 + Math.min(1, maxContribution * 6) * 0.22;

  return (
    <group position={[position.x, position.y, position.z]}>
      <mesh
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover(id);
        }}
        onPointerOut={() => onHover(null)}
      >
        <sphereGeometry args={[radius, 24, 24]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isFocus ? 0.9 : hovered ? 0.5 : 0.2}
        />
      </mesh>
      <Html position={[0, radius + 0.35, 0]} center distanceFactor={12} style={{ pointerEvents: "none" }}>
        <div className={`causal-node-label${isFocus ? " causal-node-label-focus" : ""}`}>{id}</div>
      </Html>
    </group>
  );
}

function Scene({
  graph,
  onOpenEvent,
}: {
  graph: CausalGraphOut;
  onOpenEvent: (event: CausalGraphEvent) => void;
}) {
  const [hoveredEdge, setHoveredEdge] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  const positions = useMemo(
    () => computeCausalGraphLayout(graph.nodes, graph.edges, graph.ticker),
    [graph.nodes, graph.edges, graph.ticker],
  );

  const nodeMaxContribution = useMemo(() => {
    const m = new Map<string, number>();
    for (const e of graph.events) {
      for (const nodeId of e.path) {
        m.set(nodeId, Math.max(m.get(nodeId) ?? 0, e.contribution));
      }
    }
    return m;
  }, [graph.events]);

  const fog = resolveVar("var(--bg-void)");

  return (
    <>
      <color attach="background" args={[resolveVar("var(--bg-panel)")]} />
      <fog attach="fog" args={[fog, 14, 30]} />
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={80} />
      <pointLight position={[-10, -5, -10]} intensity={30} />

      <CameraRig />
      <OrbitControls enableDamping dampingFactor={0.08} autoRotate autoRotateSpeed={0.4} makeDefault />

      {graph.edges.map((edge) => {
        if (!positions[edge.src] || !positions[edge.dst]) return null;
        const event = primaryEventForEdge(graph.events, edge.src, edge.dst);
        const key = `${edge.src}->${edge.dst}->${edge.channel ?? ""}`;
        return (
          <Edge
            key={key}
            src={edge.src}
            dst={edge.dst}
            weight={edge.weight}
            positions={positions}
            event={event}
            hovered={hoveredEdge === `${edge.src}->${edge.dst}`}
            onHover={setHoveredEdge}
            onClick={() => {
              if (event) onOpenEvent(event);
            }}
          />
        );
      })}

      {graph.nodes.map((n) => {
        const pos = positions[n.id];
        if (!pos) return null;
        return (
          <Node
            key={n.id}
            id={n.id}
            nodeType={n.node_type}
            isFocus={n.is_focus}
            position={pos}
            maxContribution={nodeMaxContribution.get(n.id) ?? 0}
            hovered={hoveredNode === n.id}
            onHover={setHoveredNode}
          />
        );
      })}
    </>
  );
}

export function CausalityScene3D({ graph, onSelectEvent }: Props) {
  const [selectedEvent, setSelectedEvent] = useState<CausalGraphEvent | null>(null);

  const legendTypes = useMemo(() => {
    const present = new Set(graph.events.map((e) => e.event_type));
    return EVENT_TYPES.filter((t) => present.has(t));
  }, [graph.events]);

  return (
    <div className="causality-scene">
      <Canvas camera={{ position: [0, 16, 24], fov: 45 }} gl={{ antialias: true }}>
        <Scene graph={graph} onOpenEvent={setSelectedEvent} />
      </Canvas>

      {selectedEvent && (
        <div className="causal-node-card">
          <button className="causal-node-card-close" onClick={() => setSelectedEvent(null)} aria-label="Close">
            &times;
          </button>
          <div className="causal-node-card-type">{selectedEvent.event_type.replaceAll("_", " ")}</div>
          <div className="causal-node-card-title">{selectedEvent.title ?? "(no headline available)"}</div>
          <p className="causal-node-card-narrative">
            {selectedEvent.narrative ?? "AI explanation not yet generated for this event."}
          </p>
          <button className="causal-node-card-link" onClick={() => onSelectEvent(selectedEvent.event_id)}>
            View full event &rarr;
          </button>
        </div>
      )}

      <ul className="chart-legend">
        {legendTypes.map((t) => (
          <li key={t}>
            <span className="legend-swatch" style={{ background: EVENT_TYPE_COLOR[t] }} />
            {t.replaceAll("_", " ")}
          </li>
        ))}
      </ul>
      <p className="chart-hint">
        Drag to orbit, scroll to zoom. Node size scales with an event's contribution; edge thickness and pulse
        speed scale with the exposure channel's transmission weight. Click an edge for its AI-generated
        explanation.
      </p>
    </div>
  );
}
