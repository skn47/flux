import type { SectorSummary } from "../types";

interface Props {
  summary: SectorSummary;
}

export function SectorBacktestTable({ summary }: Props) {
  const { per_sector, hits, n_sectors } = summary.flux_vs_baseline_hit_rate;

  return (
    <div>
      <p className="headline-finding">
        Flux beats baseline in {hits} of {n_sectors} sectors tested.
      </p>
      {summary.dataset_version_note && (
        <p className="chart-hint">{summary.dataset_version_note}</p>
      )}
      <table className="prediction-table">
        <thead>
          <tr>
            <th>Sector</th>
            <th>Baseline Sharpe</th>
            <th>Flux Sharpe</th>
            <th>Flux beats baseline?</th>
          </tr>
        </thead>
        <tbody>
          {per_sector.map((row) => (
            <tr key={row.sector}>
              <td>{row.sector}</td>
              <td className="num">{row.baseline_sharpe.toFixed(3)}</td>
              <td className="num">{row.flux_sharpe.toFixed(3)}</td>
              <td className={`num ${row.flux_beats_baseline ? "yes" : "no"}`}>
                {row.flux_beats_baseline ? "Yes" : "No"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
