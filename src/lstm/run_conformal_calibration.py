"""
Entry point: Direction 4 -- conformal calibration of the seq2seq model's
MC-Dropout uncertainty, via walk-forward. Motivation (measured, not
hypothetical): the raw MC-Dropout 90%-nominal band on the production
`none`/`concat` checkpoints achieves only ~5.2%/~4.2% empirical coverage
(backtest/seq2seq_backtest_summary.json) -- badly overconfident. This script
fixes that using CQR (lstm/conformal.py) calibrated on genuinely out-of-fold
nonconformity scores from R5-style walk-forward folds (backtest/
seq_walk_forward.py), not a single contiguous calibration block.

Pipeline (mirrors the train/val/test discipline used everywhere else in this
project, one level up at fold granularity):

  1. Compute 4 expanding-window folds (fold_id 101-104) and train the
     seq2seq model FRESH on each fold x {"none", "concat"} -- 8 full
     training runs, never warm-started (backtest.seq_walk_forward).
  2. Generate each fold's own held-out full-path MC-Dropout predictions
     (backtest.seq_predictions.generate_full_path_predictions_seq) --
     genuinely out-of-sample for that fold's own model.
  3. Fit per-step CQR quantiles (lstm.conformal.fit_cqr_calibration) from
     folds 101-103's POOLED nonconformity scores only.
  4. HELD-OUT CHECK: apply that calibration to fold 104's predictions (fold
     104 was never used to fit q_hat) and report empirical coverage -- the
     honest generalization number, not a circular self-check.
  5. Also apply the same calibration to the already-shipped production
     `lstm/models/seq2seq/{none,concat}.pt` checkpoints' existing
     single-split TEST predictions, explicitly flagged as a distinct
     assumption (calibration fit on walk-forward fold models, applied to a
     separately-trained production model).

Writes `lstm/models/seq2seq/conformal_{variant}.json` (the calibration
itself, reusable at inference without rerunning this whole pipeline) and
`lstm/conformal_calibration_report.json` (the full before/after numbers).

Run (expect this to take a while -- 8 full seq2seq training runs):
    ./.venv/bin/python -m lstm.run_conformal_calibration
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.backtest.seq_calibration import compute_coverage_table, compute_coverage_table_calibrated
from src.backtest.seq_predictions import generate_full_path_predictions_seq
from src.backtest.seq_walk_forward import DEFAULT_SEQ_WF_VARIANTS, train_all_folds
from src.flux_engine.projection import FluxTrajectoryProjector
from src.lstm.conformal import cqr_nonconformity_scores, fit_cqr_calibration, save_calibration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "lstm" / "models" / "seq2seq"

ALPHA = 0.10
LO_PCT = 5.0
HI_PCT = 95.0


def _merge_scores(score_dicts: list[dict[int, list[float]]]) -> dict[int, list[float]]:
    merged: dict[int, list[float]] = {}
    for d in score_dicts:
        for k, v in d.items():
            merged.setdefault(k, []).extend(v)
    return merged


def _coverage_summary(table: list[dict]) -> dict:
    vals = [r["empirical_coverage_pct"] for r in table]
    return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}


def main():
    variants = DEFAULT_SEQ_WF_VARIANTS

    print("Step 1/4: computing walk-forward fold boundaries and training all folds x variants "
          "(fresh from scratch, never warm-started)...")
    folds, train_results = train_all_folds(variants=variants)
    ckpt_by_fold_variant = {(r["fold_id"], r["variant"]): r["checkpoint_path"] for r in train_results}

    calibration_fold_ids = [f["fold_id"] for f in folds[:-1]]  # folds 101-103
    heldout_fold_id = folds[-1]["fold_id"]                      # fold 104, NEVER used to fit q_hat
    print(f"\nCalibration folds: {calibration_fold_ids}  Held-out (verification) fold: {heldout_fold_id}")

    projector = FluxTrajectoryProjector()

    print("\nStep 2/4: generating full-path MC-Dropout predictions per fold/variant "
          "on each fold's OWN held-out test window...")
    preds_by_fold_variant = {}
    for fold in folds:
        fold_id = fold["fold_id"]
        split_boundaries = {k: fold[k] for k in ("train", "val", "test")}
        for variant in variants:
            ckpt_path = Path(ckpt_by_fold_variant[(fold_id, variant)])
            preds = generate_full_path_predictions_seq(
                variant, checkpoint_path=ckpt_path, split_boundaries=split_boundaries, projector=projector,
            )
            preds_by_fold_variant[(fold_id, variant)] = preds
            print(f"  fold {fold_id}/{variant}: {len(preds)} test-window predictions")

    print(f"\nStep 3/4: fitting CQR calibration (alpha={ALPHA}) on folds {calibration_fold_ids}, "
          f"verifying on held-out fold {heldout_fold_id}...")
    calibrations = {}
    heldout_report = {}
    for variant in variants:
        pooled_scores = _merge_scores([
            cqr_nonconformity_scores(preds_by_fold_variant[(fid, variant)], lo_pct=LO_PCT, hi_pct=HI_PCT)
            for fid in calibration_fold_ids
        ])
        calibration = fit_cqr_calibration(
            pooled_scores, alpha=ALPHA, lo_pct=LO_PCT, hi_pct=HI_PCT, variant=variant,
            fold_ids_used=calibration_fold_ids,
        )
        calibrations[variant] = calibration
        save_calibration(calibration, MODELS_DIR / f"conformal_{variant}.json")
        print(f"  [{variant}] q_hat (steps 1-5 of {calibration.horizon}): "
              f"{[round(q, 5) for q in calibration.q_hat[:5]]}  "
              f"n_cal_per_step (steps 1-5): {calibration.n_cal_per_step[:5]}")

        heldout_preds = preds_by_fold_variant[(heldout_fold_id, variant)]
        raw_table = compute_coverage_table(heldout_preds, lo_pct=LO_PCT, hi_pct=HI_PCT)
        cal_table = compute_coverage_table_calibrated(heldout_preds, calibration)
        heldout_report[variant] = {
            "raw": raw_table, "calibrated": cal_table,
            "raw_summary": _coverage_summary(raw_table),
            "calibrated_summary": _coverage_summary(cal_table),
            "n_windows": len(heldout_preds),
        }
        print(f"  [{variant}] HELD-OUT fold {heldout_fold_id} coverage: "
              f"raw mean={heldout_report[variant]['raw_summary']['mean']:.2f}%  "
              f"calibrated mean={heldout_report[variant]['calibrated_summary']['mean']:.2f}%  "
              f"(nominal {100 * (1 - ALPHA):.1f}%)")

    print("\nStep 4/4: applying calibration to the PRODUCTION single-split checkpoints' "
          "existing test-split predictions (calibration fit on walk-forward FOLD models, "
          "applied here to the separately-trained production model -- a distinct assumption, "
          "stated explicitly)...")
    production_report = {}
    for variant in variants:
        prod_preds = generate_full_path_predictions_seq(variant, projector=projector)
        raw_table = compute_coverage_table(prod_preds, lo_pct=LO_PCT, hi_pct=HI_PCT)
        cal_table = compute_coverage_table_calibrated(prod_preds, calibrations[variant])
        production_report[variant] = {
            "raw": raw_table, "calibrated": cal_table,
            "raw_summary": _coverage_summary(raw_table),
            "calibrated_summary": _coverage_summary(cal_table),
            "n_windows": len(prod_preds),
        }
        print(f"  [{variant}] PRODUCTION checkpoint coverage: "
              f"raw mean={production_report[variant]['raw_summary']['mean']:.2f}%  "
              f"calibrated mean={production_report[variant]['calibrated_summary']['mean']:.2f}%  "
              f"(nominal {100 * (1 - ALPHA):.1f}%)")

    report = {
        "alpha": ALPHA, "lo_pct": LO_PCT, "hi_pct": HI_PCT,
        "folds": folds,
        "calibration_fold_ids": calibration_fold_ids,
        "heldout_fold_id": heldout_fold_id,
        "train_results": train_results,
        "calibrations": {v: c.to_dict() for v, c in calibrations.items()},
        "heldout_report": heldout_report,
        "production_report": production_report,
    }
    out_path = REPO_ROOT / "lstm" / "conformal_calibration_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {out_path}")
    print(f"Calibration artifacts: {MODELS_DIR}/conformal_none.json, {MODELS_DIR}/conformal_concat.json")


if __name__ == "__main__":
    main()
