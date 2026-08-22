"""
Entry point: trains BOTH feature variants (baseline, flux-augmented) with
identical architecture/hyperparameters, evaluates both on train/val/test,
and reports MAE/RMSE/MAPE per ticker and averaged -- per task spec point 6.

R1 REWORK (see lstm/dataset.py's module docstring): target is now
`daily_return`, so MAE/RMSE below are on raw RETURN units, not raw price
units, and MAE/RMSE (not MAPE) are the primary headline metrics -- see
lstm/evaluate.py's module docstring for why MAPE is noisier on a return
target. Also reports two model-free persistence baselines ("predict return =
0", "predict return = yesterday's return") computed directly from raw test
data, so a reader can see how much (if anything) the LSTM earns over trivial
baselines (`lstm.evaluate.persistence_baseline_metrics`).

No backtesting / trading-signal conversion / Sharpe / drawdown here (task
spec point 7) -- stops at trained models + prediction-accuracy metrics.

Run:
    ./.venv/bin/python -m lstm.run_train
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from config.mvp_scope import TRACKED_TICKERS
from src.lstm.dataset import (
    DEFAULT_DB_PATH,
    LOOKBACK,
    TARGET_COL,
    VARIANTS,
    Window,
    build_pooled_dataset,
    load_ticker_frame,
    verify_no_leakage,
)
from src.lstm.evaluate import (
    assert_metrics_agree,
    compute_metrics,
    persistence_baseline_metrics,
    recompute_metrics_independently,
)
from src.lstm.train import TrainConfig, get_device, set_seed, train_model

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "lstm" / "models"


def predict_scaled(model, windows: list[Window], device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(windows), batch_size):
            batch = windows[i: i + batch_size]
            X = torch.tensor(np.stack([w.X for w in batch]), dtype=torch.float32).to(device)
            ticker_ids = torch.tensor([w.ticker_id for w in batch], dtype=torch.long).to(device)
            out = model(X, ticker_ids).cpu().numpy()
            preds.append(out)
    return np.concatenate(preds) if preds else np.array([])


def evaluate_split(
    model, windows: list[Window], scalers: dict, feature_cols: list[str], device
) -> dict:
    """Per-ticker (and averaged) MAE/RMSE/MAPE on raw adj_close units,
    computed by inverse-transforming each window's scaled prediction with
    its OWN ticker's train-fit scaler."""
    target_idx = feature_cols.index(TARGET_COL)
    by_ticker: dict[str, list[Window]] = {}
    for w in windows:
        by_ticker.setdefault(w.ticker, []).append(w)

    per_ticker_metrics = {}
    for ticker, wins in by_ticker.items():
        preds_scaled = predict_scaled(model, wins, device)
        preds_raw = scalers[ticker].inverse_transform_col(preds_scaled, target_idx)
        actual_raw = np.array([w.y_raw for w in wins])

        m_vec = compute_metrics(actual_raw, preds_raw)
        m_indep = recompute_metrics_independently(actual_raw.tolist(), preds_raw.tolist())
        assert_metrics_agree(m_vec, m_indep)  # independent re-derivation check
        per_ticker_metrics[ticker] = m_vec

    if per_ticker_metrics:
        avg = {
            "n": sum(m["n"] for m in per_ticker_metrics.values()),
            "mae": float(np.mean([m["mae"] for m in per_ticker_metrics.values()])),
            "rmse": float(np.mean([m["rmse"] for m in per_ticker_metrics.values()])),
            "mape": float(np.mean([m["mape"] for m in per_ticker_metrics.values()])),
        }
    else:
        avg = {"n": 0, "mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}
    return {"per_ticker": per_ticker_metrics, "average": avg}


def compute_persistence_baseline(
    windows: list[Window], db_path: Path | str = DEFAULT_DB_PATH
) -> dict:
    """
    Model-free "predict return = 0" / "predict return = yesterday's return"
    baselines (R1, see lstm/evaluate.py::persistence_baseline_metrics),
    computed directly from RAW (unscaled) daily_return data -- identical
    regardless of feature variant, since baseline/flux share the same
    windows/labels/split (only the feature vector differs between them).

    For each window (label date L), y_true = w.y_raw (raw daily_return at
    L); y_prior = raw daily_return at L-1 (w.feature_end_date), fetched
    fresh per ticker via lstm.dataset.load_ticker_frame -- deliberately NOT
    reconstructed by inverse-transforming the window's SCALED feature
    matrix, so this baseline has zero dependency on any scaler/model code.
    """
    raw_return_by_ticker_date: dict[str, dict[str, float]] = {}
    y_true: list[float] = []
    y_prior: list[float] = []
    for w in windows:
        if w.ticker not in raw_return_by_ticker_date:
            frame = load_ticker_frame(w.ticker, db_path=db_path)
            raw_return_by_ticker_date[w.ticker] = dict(zip(frame["date"], frame["daily_return"]))
        y_true.append(w.y_raw)
        y_prior.append(raw_return_by_ticker_date[w.ticker][w.feature_end_date])
    return persistence_baseline_metrics(np.array(y_true), np.array(y_prior))


def run_variant(variant: str, config: TrainConfig) -> dict:
    print(f"\n{'=' * 70}\nVariant: {variant}  (features: {VARIANTS[variant]})\n{'=' * 70}")
    set_seed(config.seed)
    device = get_device()
    print(f"Device: {device}")

    by_split, scalers = build_pooled_dataset(variant, tickers=TRACKED_TICKERS, lookback=LOOKBACK)
    for split_name in ("train", "val", "test"):
        print(f"  {split_name}: {len(by_split[split_name])} windows")

    leakage_report = verify_no_leakage(by_split)
    print(f"  Leakage check: {leakage_report}")

    n_features = len(VARIANTS[variant])
    n_tickers = len(TRACKED_TICKERS)

    result = train_model(
        train_windows=by_split["train"],
        val_windows=by_split["val"],
        n_features=n_features,
        n_tickers=n_tickers,
        config=config,
        device=device,
        verbose=True,
    )
    print(f"  Best epoch: {result.best_epoch}  best val MSE (scaled): {result.best_val_loss:.6f}")

    metrics = {}
    for split_name in ("train", "val", "test"):
        metrics[split_name] = evaluate_split(
            result.model, by_split[split_name], scalers, VARIANTS[variant], device
        )
        avg = metrics[split_name]["average"]
        print(
            f"  [{variant}/{split_name}] avg over {len(scalers)} tickers: "
            f"MAE={avg['mae']:.4f} RMSE={avg['rmse']:.4f} MAPE={avg['mape']:.3f}%"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / f"{variant}.pt"
    torch.save(
        {
            "model_state": result.model.state_dict(),
            "variant": variant,
            "feature_cols": VARIANTS[variant],
            "n_features": n_features,
            "n_tickers": n_tickers,
            "ticker_order": TRACKED_TICKERS,
            "lookback": LOOKBACK,
            "config": config.__dict__,
            "best_epoch": result.best_epoch,
            "best_val_loss_scaled_mse": result.best_val_loss,
            "scalers": {t: s.to_dict() for t, s in scalers.items()},
        },
        ckpt_path,
    )
    print(f"  Saved checkpoint: {ckpt_path}")

    history_path = MODELS_DIR / f"{variant}_history.json"
    with open(history_path, "w") as f:
        json.dump(result.history, f, indent=2)

    metrics_path = MODELS_DIR / f"{variant}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "variant": variant,
        "best_epoch": result.best_epoch,
        "best_val_loss_scaled_mse": result.best_val_loss,
        "n_epochs_trained": len(result.history),
        "metrics": metrics,
        "leakage_report": leakage_report,
        "window_counts": {k: len(v) for k, v in by_split.items()},
    }


def main():
    config = TrainConfig()
    all_results = {}
    for variant in ("baseline", "flux"):
        all_results[variant] = run_variant(variant, config)

    # Persistence baselines (R1, model-free) -- computed once, not per
    # variant, since baseline/flux share identical windows/labels/split
    # (task spec guarantee, restated in lstm/dataset.py's module docstring);
    # rebuilding via the "baseline" variant's windows is representative of
    # both (only the feature vector X differs between variants, not y_raw/
    # feature_end_date/ticker/label_date, which is all this baseline needs).
    print(f"\n{'=' * 70}\nPersistence baselines (model-free, raw daily_return units)\n{'=' * 70}")
    by_split_for_baseline, _ = build_pooled_dataset("baseline", tickers=TRACKED_TICKERS, lookback=LOOKBACK)
    persistence_by_split = {}
    for split_name in ("train", "val", "test"):
        pb = compute_persistence_baseline(by_split_for_baseline[split_name])
        persistence_by_split[split_name] = pb
        print(
            f"  [{split_name}] zero-baseline : MAE={pb['zero']['mae']:.6f} RMSE={pb['zero']['rmse']:.6f} "
            f"MAPE={pb['zero']['mape']:.3f}%"
        )
        print(
            f"  [{split_name}] prior-baseline: MAE={pb['prior']['mae']:.6f} RMSE={pb['prior']['rmse']:.6f} "
            f"MAPE={pb['prior']['mape']:.3f}%"
        )

    all_results["persistence_baselines"] = persistence_by_split

    summary_path = MODELS_DIR / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull run summary written to {summary_path}")

    print(f"\n{'=' * 70}\nTest-set comparison (baseline vs. flux-augmented vs. persistence)\n{'=' * 70}")
    b_test = all_results["baseline"]["metrics"]["test"]["average"]
    s_test = all_results["flux"]["metrics"]["test"]["average"]
    p_zero = persistence_by_split["test"]["zero"]
    p_prior = persistence_by_split["test"]["prior"]
    print(f"  baseline      : MAE={b_test['mae']:.6f} RMSE={b_test['rmse']:.6f} MAPE={b_test['mape']:.3f}%")
    print(f"  flux         : MAE={s_test['mae']:.6f} RMSE={s_test['rmse']:.6f} MAPE={s_test['mape']:.3f}%")
    print(f"  persist(zero) : MAE={p_zero['mae']:.6f} RMSE={p_zero['rmse']:.6f} MAPE={p_zero['mape']:.3f}%")
    print(f"  persist(prior): MAE={p_prior['mae']:.6f} RMSE={p_prior['rmse']:.6f} MAPE={p_prior['mape']:.3f}%")
    mae_delta = s_test["mae"] - b_test["mae"]
    verdict = "HELPED" if mae_delta < 0 else ("HURT" if mae_delta > 0 else "NO CHANGE")
    print(f"  MAE delta (flux - baseline): {mae_delta:+.6f} -> flux_score {verdict} (primary metric, R1)")
    mape_delta = s_test["mape"] - b_test["mape"]
    if np.isnan(mape_delta):
        print(
            f"  MAPE delta (flux - baseline): UNDEFINED (at least one side is inf/nan -- "
            f"a near/exact-zero-return test day dominates the divide-by-y_true MAPE formula; "
            f"see lstm/evaluate.py's module docstring and lstm/README.md's MAPE-on-returns "
            f"caveat -- MAE above is the primary metric for exactly this reason)"
        )
    else:
        mape_verdict = "HELPED" if mape_delta < 0 else ("HURT" if mape_delta > 0 else "NO CHANGE")
        print(f"  MAPE delta (flux - baseline): {mape_delta:+.4f} pp -> flux_score {mape_verdict} (secondary, noisy on returns)")


if __name__ == "__main__":
    main()
