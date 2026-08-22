import { useEffect, useState } from "react";
import { EventImpactChart } from "./EventImpactChart";
import { Modal } from "./Modal";
import { api } from "../services/api";
import { SECTOR_COLOR } from "../lib/sectorColors";
import { severityBand } from "../lib/severityBands";
import { impactTier, contributionTier } from "../lib/exposureBands";
import type { EventDetail } from "../types";

interface Props {
  eventId: string;
  onClose: () => void;
}

// Ex-pages/EventDetail.tsx, now an in-page modal instead of a route (opened
// from any OSINT headline or drilldown item via ?event=).
//
// Deliberately shows only what a non-technical reader needs: headline,
// classification, source, affected tickers, and severity/impact/contribution
// as plain-language bands -- not the full internal record (confidence,
// polarity, k-value, countries, cascade path, etc.), which stays available in
// the underlying EventDetail/EventTickerImpact types for other consumers
// (e.g. WhatIfDrawer.tsx) even though this view no longer renders it.
export function EventModal({ eventId, onClose }: Props) {
  const [event, setEvent] = useState<EventDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setEvent(null);
    setError(null);
    api
      .eventDetail(eventId)
      .then(setEvent)
      .catch((e) => setError(String(e)));
  }, [eventId]);

  return (
    <Modal title={event ? event.title ?? "(untitled event)" : "Event"} onClose={onClose}>
      {error && (
        <p className="error">
          This event isn't currently available — it may have been filtered out or superseded.
        </p>
      )}
      {!error && !event && <p>Loading…</p>}
      {event && (
        <div>
          <table className="prediction-table">
            <tbody>
              <tr>
                <td>Classification</td>
                <td>{event.event_type.replaceAll("_", " ")}</td>
              </tr>
              <tr>
                <td>Severity</td>
                <td title={`severity_score ${event.severity_score.toFixed(3)}`}>
                  {severityBand(event.severity_score).name}
                </td>
              </tr>
              <tr>
                <td>Source</td>
                <td>
                  {event.url ? (
                    <a className="source-link" href={event.url} target="_blank" rel="noreferrer">
                      Read source article ↗
                    </a>
                  ) : (
                    "No source article on file."
                  )}
                </td>
              </tr>
            </tbody>
          </table>

          <section>
            <h3>Affected tickers</h3>
            {event.affected.length === 0 ? (
              <p>No tracked tickers were reached by this event.</p>
            ) : (
              <>
                <EventImpactChart affected={event.affected} />
                <table className="prediction-table">
                  <thead>
                    <tr>
                      <th>Ticker</th>
                      <th>Sector</th>
                      <th>Impact</th>
                      <th>Contribution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...event.affected]
                      .sort((a, b) => b.contribution - a.contribution)
                      .map((row) => {
                        const imp = impactTier(row.imp);
                        const contrib = contributionTier(row.contribution);
                        return (
                          <tr key={row.ticker}>
                            <td className="num">{row.ticker}</td>
                            <td>
                              <span
                                className="ticker-tag"
                                style={{ borderColor: SECTOR_COLOR[row.sector] }}
                              >
                                {row.sector.replaceAll("_", " ")}
                              </span>
                            </td>
                            <td title={`imp ${row.imp.toFixed(4)}`}>
                              <span className="tier-badge" style={{ color: imp.colorVar, borderColor: imp.colorVar }}>
                                {imp.label}
                              </span>
                            </td>
                            <td title={`contribution ${row.contribution.toFixed(4)}`}>
                              <span
                                className="tier-badge"
                                style={{ color: contrib.colorVar, borderColor: contrib.colorVar }}
                              >
                                {contrib.label}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </>
            )}
          </section>
        </div>
      )}
    </Modal>
  );
}
