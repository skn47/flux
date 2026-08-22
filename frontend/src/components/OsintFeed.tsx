import type { EventSummary } from "../types";

interface Props {
  events: EventSummary[];
  onSelectEvent: (eventId: string) => void;
}

// Left-column OSINT feed. Replaces both Terminal.tsx's old inline "Recent
// events" panel (which truncated to 12) and the standalone Events.tsx route
// -- this is now the single, always-visible, untruncated feed, scrollable
// within its column instead of being its own page.
export function OsintFeed({ events, onSelectEvent }: Props) {
  return (
    <div className="panel osint-feed">
      <div className="panel-header">OSINT feed</div>
      {events.length === 0 ? (
        <p>Loading…</p>
      ) : (
        <ul className="event-list osint-list">
          {events.map((e) => (
            <li key={e.event_id} className="event-item">
              <button className="event-headline-link" onClick={() => onSelectEvent(e.event_id)}>
                {e.title ?? (
                  <>
                    <span className="event-type">{e.event_type.replaceAll("_", " ")}</span> in{" "}
                    <span className="event-corridor">{e.corridor.join(", ")}</span>
                  </>
                )}
              </button>
              {e.title && (
                <span className="badge-note" title={e.corridor.join(", ")}>
                  {e.event_type.replaceAll("_", " ")}
                </span>
              )}
              <div className="event-meta">
                {new Date(e.published_at).toLocaleDateString()} &middot; severity{" "}
                {e.severity_score.toFixed(2)}
                {e.title && <> &middot; {e.corridor.join(", ")}</>}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
