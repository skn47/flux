"""
Entry point: trains the seq2seq 30-day-forward model, per the recommended
rollout in lstm/seq2seq_README.md -- `none` (no flux anywhere, analogous to
PriceLSTM's baseline) and `concat` (flux_score in the encoder's historical
window AND per-step in the decoder) by default; `film` (FiLM-gated decoder
conditioning) exists and is fully wired but deliberately NOT trained by
default here -- add it to DEFAULT_SEQ_VARIANTS once `none`/`concat` are
confirmed to train stably and beat the multi-step persistence baselines.

Mirrors lstm/run_train.py's structure and reporting conventions (per-ticker
+ averaged metrics, persistence-baseline comparison, HELPED/HURT verdict,
checkpoint + history + metrics JSON artifacts) but reports PER-HORIZON-STEP
metrics (lstm.seq_evaluate.compute_metrics_per_step), not a single number,
since error is expected to grow with horizon distance.

This module trains and evaluates the model ONLY -- no backtesting / trading-
signal conversion here, same scope boundary lstm/run_train.py draws (see
backtest/run_seq2seq_backtest.py for that separate step).

Run:
    ./.venv/bin/python -m lstm.run_train_seq2seq
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from config.mvp_scope import TRACKED_TICKERS
from src.flux_engine.projection import FluxTrajectoryProjector
from src.lstm.dataset import LOOKBACK, TARGET_COL, VARIANTS
from src.lstm.seq_dataset import SeqWindow, build_pooled_seq_dataset, verify_no_leakage_seq
from src.lstm.seq_evaluate import compute_metrics_per_step, compute_persistence_baseline_seq
from src.lstm.train_seq2seq import Seq2SeqTrainConfig, get_device, set_seed, train_model

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "lstm" / "models" / "seq2seq"

# flux_conditioning -> which lstm.dataset.VARIANTS feature set the ENCODER
# sees. "none" mirrors PriceLSTM's baseline (no flux_score anywhere);
# "concat"/"film" mirror PriceLSTM's flux variant for the encoder's
# historical window (flux_score included as one of 5 input features) AND
# additionally condition the decoder on the per-step PROJECTED flux
# trajectory -- see lstm/seq2seq_model.py's module docstring.
_DATASET_VARIANT_FOR_FLUX_CONDITIONING = {"none": "baseline", "concat": "flux", "film": "flux"}

DEFAULT_SEQ_VARIANTS = ("none", "concat")  # "film" deferred -- see module docstring


def predict_scaled(model, windows: list[SeqWindow], device) -> np.ndarray:
    """(n_windows, horizon) scaled predictions. Deterministic: dropout OFF
    (model.eval()), fully autoregressive (teacher_forcing=None) -- this is
    the point-accuracy evaluation path. MC-Dropout uncertainty sampling
    (lstm.train_seq2seq.sample_trajectories, dropout ON) is a deliberately
    separate code path used only for calibration diagnostics
    (backtest/seq_calibration.py), not for these point-accuracy metrics."""
    model.eval()
    preds = []
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(windows), batch_size):
            batch = windows[i: i + batch_size]
            X = torch.tensor(np.stack([w.X for w in batch]), dtype=torch.float32).to(device)
            ticker_ids = torch.tensor([w.ticker_id for w in batch], dtype=torch.long).to(device)
            flux_traj = None
            if model.flux_conditioning != "none":
                flux_traj = torch.tensor(np.stack([w.flux_traj for w in batch]), dtype=torch.float32).to(device)
            out = model(X, ticker_ids, flux_traj=flux_traj, teacher_forcing=None).cpu().numpy()
            preds.append(out)
    return np.concatenate(preds) if preds else np.zeros((0, model.horizon))


def evaluate_split(model, windows: list[SeqWindow], scalers: dict, feature_cols: list[str], device) -> dict:
    """Per-ticker (and averaged) per-horizon-step MAE/RMSE/MAPE on raw
    daily_return units, inverse-transforming each window's scaled prediction
    PATH with its own ticker's train-fit scaler, one horizon step at a
    time (StandardScaler.inverse_transform_col operates on a 1D column)."""
    target_idx = feature_cols.index(TARGET_COL)
    by_ticker: dict[str, list[SeqWindow]] = {}
    for w in windows:
        by_ticker.setdefault(w.ticker, []).append(w)

    per_ticker_metrics = {}
    for ticker, wins in by_ticker.items():
        preds_scaled = predict_scaled(model, wins, device)  # (n, horizon)
        preds_raw = np.stack(
            [scalers[ticker].inverse_transform_col(preds_scaled[:, k], target_idx)
             for k in range(preds_scaled.shape[1])],
            axis=1,
        )
        actual_raw = np.stack([w.y_raw for w in wins])
        per_ticker_metrics[ticker] = compute_metrics_per_step(actual_raw, preds_raw)

    if per_ticker_metrics:
        avg = {
            "n": sum(m["all_steps"]["n"] for m in per_ticker_metrics.values()),
            "mae": float(np.mean([m["all_steps"]["mae"] for m in per_ticker_metrics.values()])),
            "rmse": float(np.mean([m["all_steps"]["rmse"] for m in per_ticker_metrics.values()])),
            "mape": float(np.mean([m["all_steps"]["mape"] for m in per_ticker_metrics.values()])),
        }
        # step-1 (next-day) slice averaged across tickers -- the number most
        # directly comparable to lstm/run_train.py's 1-day-ahead headline,
        # since backtest/seq_predictions.py's k=1 integration (see
        # lstm/seq2seq_README.md) trades off exactly this slice.
        avg_step1 = {
            "mae": float(np.mean([m["per_step"][0]["mae"] for m in per_ticker_metrics.values()])),
            "rmse": float(np.mean([m["per_step"][0]["rmse"] for m in per_ticker_metrics.values()])),
            "mape": float(np.mean([m["per_step"][0]["mape"] for m in per_ticker_metrics.values()])),
        }
    else:
        avg = {"n": 0, "mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}
        avg_step1 = {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}
    return {"per_ticker": per_ticker_metrics, "average": avg, "average_step1": avg_step1}


def run_variant(variant: str, base_config: Seq2SeqTrainConfig, projector: FluxTrajectoryProjector) -> dict:
    flux_conditioning = variant
    dataset_variant = _DATASET_VARIANT_FOR_FLUX_CONDITIONING[flux_conditioning]
    feature_cols = VARIANTS[dataset_variant]
    config = replace(base_config, flux_conditioning=flux_conditioning)

    print(f"\n{'=' * 70}\nSeq2seq variant: {variant}  "
          f"(encoder features: {feature_cols}, flux_conditioning={flux_conditioning})\n{'=' * 70}")
    set_seed(config.seed)
    device = get_device()
    print(f"Device: {device}")

    by_split, scalers = build_pooled_seq_dataset(
        dataset_variant, tickers=TRACKED_TICKERS, lookback=LOOKBACK, horizon=config.horizon,
        projector=projector,
    )
    for split_name in ("train", "val", "test"):
        print(f"  {split_name}: {len(by_split[split_name])} windows")

    leakage_report = verify_no_leakage_seq(by_split)
    print(f"  Leakage check: {leakage_report}")

    n_features = len(feature_cols)
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
            result.model, by_split[split_name], scalers, feature_cols, device
        )
        avg = metrics[split_name]["average"]
        avg1 = metrics[split_name]["average_step1"]
        print(
            f"  [{variant}/{split_name}] all-steps avg over {len(scalers)} tickers: "
            f"MAE={avg['mae']:.4f} RMSE={avg['rmse']:.4f} MAPE={avg['mape']:.3f}% "
            f"| step-1 only: MAE={avg1['mae']:.4f} RMSE={avg1['rmse']:.4f}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / f"{variant}.pt"
    torch.save(
        {
            "model_state": result.model.state_dict(),
            "variant": variant,
            "flux_conditioning": flux_conditioning,
            "horizon": config.horizon,
            "feature_cols": feature_cols,
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
    config = Seq2SeqTrainConfig()
    # One shared FluxTrajectoryProjector (one global cluster pass) reused
    # across every variant/ticker -- see flux_engine/projection.py's
    # docstring for why rebuilding this per call would be wasteful.
    projector = FluxTrajectoryProjector()

    all_results = {}
    for variant in DEFAULT_SEQ_VARIANTS:
        all_results[variant] = run_variant(variant, config, projector)

    print(f"\n{'=' * 70}\nMulti-step persistence baselines "
          f"(model-free, raw daily_return units)\n{'=' * 70}")
    by_split_for_baseline, _ = build_pooled_seq_dataset(
        "baseline", tickers=TRACKED_TICKERS, lookback=LOOKBACK, horizon=config.horizon, projector=projector,
    )
    persistence_by_split = {}
    for split_name in ("train", "val", "test"):
        pb = compute_persistence_baseline_seq(by_split_for_baseline[split_name])
        persistence_by_split[split_name] = pb
        z, p = pb["zero"]["all_steps"], pb["prior"]["all_steps"]
        print(f"  [{split_name}] zero-baseline (all steps) : MAE={z['mae']:.6f} RMSE={z['rmse']:.6f}")
        print(f"  [{split_name}] prior-baseline (all steps): MAE={p['mae']:.6f} RMSE={p['rmse']:.6f}")

    all_results["persistence_baselines"] = persistence_by_split

    summary_path = MODELS_DIR / "run_summary.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull run summary written to {summary_path}")

    print(f"\n{'=' * 70}\nTest-set comparison (all-steps MAE vs. persistence)\n{'=' * 70}")
    p_zero = persistence_by_split["test"]["zero"]["all_steps"]
    p_prior = persistence_by_split["test"]["prior"]["all_steps"]
    print(f"  persist(zero) : MAE={p_zero['mae']:.6f} RMSE={p_zero['rmse']:.6f}")
    print(f"  persist(prior): MAE={p_prior['mae']:.6f} RMSE={p_prior['rmse']:.6f}")
    for variant in DEFAULT_SEQ_VARIANTS:
        v_test = all_results[variant]["metrics"]["test"]["average"]
        mae_delta = v_test["mae"] - p_zero["mae"]
        verdict = "HELPED" if mae_delta < 0 else ("HURT" if mae_delta > 0 else "NO CHANGE")
        print(
            f"  {variant:<8s}    : MAE={v_test['mae']:.6f} RMSE={v_test['rmse']:.6f} "
            f"(delta vs zero-persistence: {mae_delta:+.6f} -> {verdict})"
        )


if __name__ == "__main__":
    main()
