import type { WalkForwardResults } from "../types";

interface Props {
  results: WalkForwardResults;
}

export function WalkForwardTable({ results }: Props) {
  const { sharpe_stats, n_folds } = results.aggregate;
  const strategies = Object.keys(sharpe_stats);

  return (
    <div className="walk-forward-panel">
      <p className="scope-caption">
        Scope: {results.scope.replaceAll("_", " ")}. Semiconductors only,
        8-ticker universe, predates the 23-ticker sector broadening —
        historical finding, not re-derived live.
      </p>
      <table className="prediction-table">
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Mean Sharpe ({n_folds} folds)</th>
            <th>Median</th>
            <th>Per-fold</th>
          </tr>
        </thead>
        <tbody>
          {strategies.map((s) => {
            const stat = sharpe_stats[s];
            return (
              <tr key={s}>
                <td>{s.replaceAll("_", " ")}</td>
                <td className="num">{stat.mean.toFixed(3)}</td>
                <td className="num">{stat.median !== undefined ? stat.median.toFixed(3) : "—"}</td>
                <td className="num">{stat.per_fold.map((v) => v.toFixed(2)).join(", ")}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
