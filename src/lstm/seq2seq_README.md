# Seq2seq 30-day distributional forecast (Phase F rework — architecture broadening)

**Status:** new, runs ALONGSIDE `lstm/model.py::PriceLSTM` (Phase F's original 1-day-ahead model, untouched). Not a replacement — see "Migration strategy" below.

## Why this exists

This rework broadens Flux's ambition from next-day return prediction toward a 30-day-ahead **distributional** forecast, using the competitor project "Nexus" (`knowledge/nexus-backend-main`) as an explicit baseline to surpass, not imitate. A code-level audit of Nexus's TensorFlow LSTM (its one genuinely solid layer) found:

- A real, correctly-implemented MC-Dropout uncertainty mechanism (50 forward passes with `training=True`).
- But its 30-day output is **30 independent Dense values from one forward pass** — no day-to-day correlation, no true sequential modeling, and its causal "Composite Shock Score" is largely `hash(ticker) % N` dressed up as calibrated data, with no real supply-chain graph anywhere in its codebase.

Flux already has a real, cited exposure graph (`propagation/graph.py`) and a verified noisy-OR `flux_score` formula (`flux_engine/formula.py`) — categorically more grounded than Nexus's fabricated shock-score terms. This rework leans into that advantage on two axes:

1. **A genuinely correlated, distributional 30-day forecast** — an autoregressive encoder-decoder (seq2seq) LSTM + MC-Dropout **trajectory** sampling, so uncertainty compounds and correlates day-to-day, unlike Nexus's independent-per-day approach.
2. **The causal flux signal as the decoder's primary conditioning input**, not a diluted fifth feature column — via a leakage-safe *projection* of already-known events' future decay (`flux_engine/projection.py`), fed into the decoder at every one of its 30 steps.

Directions explicitly deferred to a later pass (not in scope here): intraday/higher-frequency prediction, and conformal-calibration of uncertainty via the walk-forward infrastructure.

## Architecture

`lstm/seq2seq_model.py::Seq2SeqPriceLSTM`:

```
Encoder (mirrors PriceLSTM for direct comparability):
    Input (batch, 60, n_features)  [4 baseline / 5 flux, same as PriceLSTM]
    -> LSTM(64, full sequence) -> Dropout(0.2)
    -> LSTM(32, final hidden state)
    -> concat with ticker Embedding(dim=4) -> Linear -> decoder h0
       (c0: separate small Linear projection of the encoder's final cell
       state -- ticker identity flows into h0 only, not c0, same "inject
       once, not every step" rationale as PriceLSTM)

Decoder (nn.LSTMCell, unrolled explicitly for horizon=30 steps -- a plain
nn.LSTM cannot branch per-step the way autoregressive feedback and flux
conditioning both require):
    step input = [previous predicted return] (+ [this step's projected
                  flux value], if flux_conditioning == "concat")
    h_raw, c = LSTMCell(step_input, (h, c))
    if flux_conditioning == "film":
        gamma, beta = FluxFiLMGenerator(flux_value)
        h = gamma * h_raw + beta          # CARRIED FORWARD to next step
    else:
        h = h_raw
    pred_k = Linear(Dropout(h))            # dropout applied to the OUTPUT
                                             # copy only, never to the
                                             # carried (h, c) state
```

Three `flux_conditioning` modes:
- `"none"` — no flux anywhere (encoder feature set = `lstm.dataset.VARIANTS["baseline"]`, 4 cols). Analogous to PriceLSTM's `baseline` variant.
- `"concat"` — flux_score is in the encoder's historical window (`VARIANTS["flux"]`, 5 cols) **and** the decoder gets the per-step projected flux value concatenated into its input at every one of the 30 steps.
- `"film"` — same encoder inputs as `concat`, but the decoder's per-step flux value instead generates a `(gamma, beta)` pair that affine-modulates the decoder's hidden state, and — critically — the **modulated** state (not the pre-modulation raw state) is what gets carried into the next step. This is the concrete mechanism that gives flux compounding architectural leverage on future dynamics, not just the current step's output — the answer to "flux as primary alpha driver, not a diluted column."

### Recommended rollout (and why)

Per the plan's explicit scope decision, this pass trains **`none` and `concat` only** (`lstm/run_train_seq2seq.py::DEFAULT_SEQ_VARIANTS`). `film` is fully implemented and unit-tested (see below) but not trained by default — it adds real complexity (an extra generator network, hidden-state gating) that's harder to debug simultaneously with a brand-new autoregressive training loop on an already-small dataset. Add it to `DEFAULT_SEQ_VARIANTS` once `none`/`concat` are confirmed to train stably and beat the multi-step persistence baselines.

Loss is scalar MSE only (`nn.MSELoss()` over the full `(batch, horizon)` output) — pinball/quantile-regression heads are deferred to a gated follow-up, contingent on the MC-Dropout bands (below) looking reasonable. Walk-forward validation (the 4-fold expanding-window scheme `backtest/walk_forward.py` already uses for PriceLSTM) is also deferred to a follow-up; this pass validates on the existing single train/val/test split.

## Autoregressive training: scheduled sampling

The decoder feeds its own previous prediction back in as input at every step, so training purely with teacher forcing (always feeding the TRUE previous return) would never expose the model to its own compounding error during training — only at inference. `lstm/train_seq2seq.py::Seq2SeqTrainConfig.teacher_forcing_ratio` decays **linearly, per example, per step** (not one coin flip per batch) from 1.0 to 0.0 over `teacher_forcing_decay_epochs` (default `round(0.6 * max_epochs)`). Early stopping is suppressed until this decay schedule completes, so a model that looks good under heavy teacher forcing can't trigger early stopping before ever training under the mostly-autoregressive regime it's actually evaluated under. **Open tuning question, not settled**: whether this suppression is the right amount — revisit if it proves too conservative in practice.

## MC-Dropout trajectory sampling

`lstm/train_seq2seq.py::sample_trajectories` leaves the model in `.train()` mode (dropout active — the same technique Nexus's TensorFlow LSTM uses, correctly) and runs N full autoregressive forward passes, each explicitly seeded (`base_seed + i`). **Verified reproducible**: two independent calls with the same `base_seed` produce byte-identical output (checked directly, see "Verification" below) — this matters because every calibration diagnostic downstream depends on it.

Because each sampled pass is autoregressive end-to-end (day k's dropout-perturbed prediction feeds day k+1's input), the N sampled 30-day paths are genuinely **correlated day-to-day** — the concrete difference from Nexus, whose MC-Dropout is real but applied to 30 independent Dense outputs from one forward pass, producing 30 independent per-day distributions rather than correlated trajectories.

## Leakage safety

### The forward-realization boundary bug (found and fixed during verification, not assumed away)

`lstm/seq_dataset.py::verify_no_leakage_seq` adds one check beyond the existing label-date check: a window's target REALIZES at `label_dates[-1]`, up to 29 trading days after `label_dates[0]` — a leakage class the 1-day model could never hit (its label date and realization date are the same day). **The first version of `build_ticker_seq_dataset` only classified windows by `label_dates[0]` and did not exclude windows whose realization crossed into the next split** — running the leakage check against the real pooled dataset caught this immediately (a TRAIN window with `label_dates[0]=2025-11-03` but `label_dates[-1]=2025-12-15`, which was *inside* val). Fixed by having `build_ticker_seq_dataset` **exclude** (not merely mislabel) any window whose `label_dates[-1]` reaches the next split's start date. Re-running the pooled build afterward: `verify_no_leakage_seq` passes cleanly across all 23 tickers, with the exclusion costing ~10% of the pre-fix train/val window counts near each boundary (expected — this is exactly what excluding ~29 boundary-adjacent trading days per ticker per boundary should cost).

### The flux trajectory projection's leakage argument

`flux_engine/projection.py::FluxTrajectoryProjector` filters events to `published_at.date() <= as_of_date` **before** clustering — an event published after `as_of_date` is structurally absent from the computation, not merely down-weighted. Verified two ways:
1. Directly, via `verify_flux_trajectory_no_future_leakage`, which independently re-derives a projected trajectory using `flux_engine.formula.flux_score` (the original, doc-verified, per-date implementation — deliberately NOT `compute_daily_series`'s two-pointer bulk path, so a bug shared by both couldn't hide). Ran against real data (NVDA, 9 target dates): **exact match to float precision**.
2. The projection is a **lower bound, not an expectation** — noisy-OR only grows as contributions are added, so any event published between `as_of_date` and a future target date (the one thing this function cannot know) can only make the true future `flux_score` higher, never lower, than what's projected. This is a known, one-directional, stated bias, not an implicit unbiasedness assumption.

## Evaluation / backtest integration scope

Per the plan's explicit decision: **minimal footprint**. `backtest/seq_predictions.py::generate_test_predictions_seq` extracts only the decoder's k=1 (t+1, next-day) slice into the exact schema `backtest/predictions.py::TestPrediction` already produces — a drop-in input to the **existing, unmodified** `backtest/strategy.py`, `backtest/engine.py`, `backtest/metrics.py`. Separately, `backtest/seq_calibration.py` provides path-level diagnostics (empirical coverage vs. nominal CI band per horizon step, per-step MAE/RMSE) using the FULL MC-Dropout trajectory ensemble (`generate_full_path_predictions_seq`) — deliberately never feeding the daily-rebalance engine, to keep "does the new architecture work" separate from "does a new 30-day-hold strategy work." A true multi-day-hold backtest (sizing/exiting using the full predicted path, potentially informed by the now-implemented conformal-calibrated band — see below) is a natural follow-up — not attempted in this pass.

## Migration strategy

Every file in this rework is new (`flux_engine/projection.py`, `lstm/seq2seq_model.py`, `lstm/seq_dataset.py`, `lstm/train_seq2seq.py`, `lstm/run_train_seq2seq.py`, `lstm/seq_evaluate.py`, `backtest/seq_predictions.py`, `backtest/seq_calibration.py`, `backtest/run_seq2seq_backtest.py`, and — added for Direction 4 — `lstm/conformal.py`, `backtest/seq_walk_forward.py`, `lstm/run_conformal_calibration.py`). **Zero modification** to `lstm/model.py`, `lstm/dataset.py`, `lstm/train.py`, `backtest/predictions.py`, `backtest/strategy.py`, `backtest/engine.py`, `backtest/walk_forward.py`, `lstm/walk_forward_bounds.py` — the currently-shipped 1-day pipeline (which `api/` depends on) is unaffected, including its existing walk-forward fold_id 1-4 rows (verified untouched after the Direction 4 run). This directly extends the project's existing baseline-vs-flux honest-comparison culture (`lstm/README.md`) to a 3-way seq2seq comparison, not a silent replacement.

## Verification performed

- **Model forward/backward pass**: all 3 `flux_conditioning` modes produce `(batch, horizon)` output and backprop cleanly (synthetic data smoke test).
- **Training loop**: scheduled-sampling teacher-forcing ratio decays as designed (checked epoch-by-epoch against the formula); a short real-data run shows train/val loss decreasing.
- **MC-Dropout reproducibility**: two calls with the same `base_seed` produce byte-identical `(n_samples, batch, horizon)` output; model's train/eval mode is correctly restored afterward.
- **Flux projection correctness + leakage**: `FluxTrajectoryProjector` construction (one global cluster pass over the full live event corpus, ~6.4M raw / ~1.6M classified non-`unclassified` events) takes ~50-95s one-time; each `.project()` call thereafter is ~1.5ms (verified: 50 calls in 0.075s). Independent re-derivation via `verify_flux_trajectory_no_future_leakage` matches to float precision on real data.
- **Full pooled dataset construction + leakage check**: ran `build_pooled_seq_dataset` across all `TRACKED_TICKERS` for both the `baseline` and `flux` feature sets against the live DB; `verify_no_leakage_seq` passes cleanly on both (6,072 train / 1,081 val / 1,035 test windows for the `flux` feature set, as of this run — **found and fixed a real boundary-exclusion bug in the process**, see above).
- **Real training run**: `lstm.run_train_seq2seq` launched end-to-end against the live DB for both `none` and `concat` variants — see `lstm/models/seq2seq/run_summary.json`. CPU-only, ~7.5s/epoch measured on the full pooled dataset; `none` early-stopped at epoch 90 (best epoch 7), `concat` early-stopped at epoch 90 (best epoch 4) — both full runs completed in well under an hour, no GPU/HPC needed.
- **Backtest integration**: `backtest/run_seq2seq_backtest.py` end-to-end against real trained checkpoints — see `backtest/seq2seq_backtest_summary.json`.

## Results (first real run against the live DB — reported honestly, not cherry-picked)

**Point-accuracy (test split, all-steps MAE, raw daily_return units):**

| | MAE | RMSE | vs. zero-persistence (0.023377 MAE) |
|---|---|---|---|
| persist(zero) | 0.023377 | 0.034582 | — |
| persist(prior) | 0.032319 | 0.046800 | worse than zero |
| `none` | 0.023393 | 0.030205 | +0.000016 → **HURT** (marginally) |
| `concat` | 0.023352 | 0.030165 | −0.000025 → **HELPED** (marginally) |

Both variants sit essentially on top of the trivial "predict return = 0" baseline — the same pattern PriceLSTM's own R1 gate check found for the 1-day model. `concat` edges out `none` by a small margin, directionally consistent with this rework's flux-as-alpha-driver hypothesis, but the margin (0.00004 MAE) is far too small to call decisive on one run against a modest-sized dataset.

**Daily-rebalance backtest (k=1 slice, POOLED across 23 tickers):** `none` Sharpe=-4.10, `concat` Sharpe=+1.16. Per-ticker Sharpes range from -6.3 to +6.8 across both variants — the same small-sample noise `backtest/metrics.py`'s own docstring warns about (here the test window is even shorter than PriceLSTM's, since the horizon-30 forward-realization exclusion further shrinks it), not a statistically reliable result. Directionally, `concat` beating `none` here is consistent with the point-accuracy result above, but one pooled Sharpe number from one run is not strong evidence on its own.

**Path-level calibration — the most important finding of this pass, and a real limitation, not a rounding error:** MC-Dropout's predicted 90% band (5th-95th percentile of the sample-path ensemble) achieves only **~5.2% (`none`) / ~4.2% (`concat`) empirical coverage**, averaged across all 30 horizon steps — the model's uncertainty bands are drastically too narrow, not approximately right. Realized returns fall outside the predicted "90% likely" range roughly 19 times out of 20, not 1 time out of 10. This holds consistently across the full horizon (min/max coverage across all 30 steps: 3.9-7.1% for `none`, 2.9-5.4% for `concat` — not just bad at one step). Per-horizon-step point-error (MAE) does grow sensibly with horizon distance (step-1 MAE 0.0220 → step-30 MAE 0.0233 for both variants), so the point predictions behave as expected even though the *uncertainty* estimate does not.

This is a known failure mode of vanilla MC-Dropout (it captures model/epistemic uncertainty only, not the aleatoric/data noise that dominates single-stock daily returns) and is exactly the kind of gap **Direction 4 (conformal calibration via the walk-forward infrastructure)**, now implemented below, exists to fix.

## Conformal calibration (Direction 4) — implemented and measured

Implements CQR (Conformalized Quantile Regression, symmetric two-sided single-`q_hat` variant; Romano/Patterson/Candès 2019) on top of the raw MC-Dropout band, calibrated via genuinely out-of-fold walk-forward scores rather than a single contiguous calibration block — the same train/val/test discipline this project uses everywhere else, one level up at fold granularity.

**Pipeline** (`lstm/conformal.py`, `backtest/seq_walk_forward.py`, `lstm/run_conformal_calibration.py`): 4 expanding-window folds (R5's scheme from `backtest/walk_forward.py`, reused with one change — the usable date range reserves `horizon`=30 trading days at the tail so every window has a full label path). Fold IDs use an offset namespace (101-104) in the shared `walk_forward_bounds` SQLite table specifically to avoid colliding with R5's existing fold_id 1-4 rows for the 1-day model — verified after the run: R5's 96 fold-1-4 rows are untouched, the new 276 fold-101-104 rows (23 tickers × 3 splits × 4 folds) coexist without overlap.

The seq2seq model is retrained fresh (never warm-started) on each of the 4 folds × `{none, concat}` = 8 full runs. Each fold's own model then generates out-of-sample full-path predictions on that fold's own test window. Per-step CQR nonconformity scores from **folds 101-103 only** (3,450 pooled scores/step) are pooled to fit `q_hat` (finite-sample-corrected order statistic, Vovk et al., alpha=0.10). **Fold 104 is held out — never used to fit `q_hat`** — and used purely to check honest, non-circular generalization.

**Held-out fold 104 coverage (n=1,150 windows, never seen during calibration fitting):**

| | raw mean | raw range | calibrated mean | calibrated range | nominal |
|---|---|---|---|---|---|
| `none` | 4.44% | 3.0–6.6% | **83.50%** | 81.9–84.9% | 90.0% |
| `concat` | 3.77% | 2.5–5.7% | **83.36%** | 81.7–85.1% | 90.0% |

**Applied to the production single-split checkpoints** (`lstm/models/seq2seq/{none,concat}.pt`, n=1,035 test-split windows) — stated explicitly as a distinct assumption: `q_hat` was fit on the walk-forward fold models, applied here to a separately-trained production model, not the model that produced the calibration scores:

| | raw mean | calibrated mean | nominal |
|---|---|---|---|
| `none` | 5.19% | **83.33%** | 90.0% |
| `concat` | 4.15% | **83.10%** | 90.0% |

`q_hat` grows mildly with horizon distance as expected (`none`: 0.0354 at step 1 → 0.0400 at step 30; `concat`: 0.0364 → 0.0395), consistent with the point-error-growth pattern already observed.

**Honest read of these numbers:** calibration closes most of the gap — raw coverage was off by a factor of ~18-24x (4-5% vs. 90% nominal), calibrated coverage lands within ~6-7 percentage points of nominal (83.1-83.5% vs. 90%) on the genuinely held-out fold, not just on the folds used to fit `q_hat`. It is not exact 90% coverage, and shouldn't be presented as such. The remaining shortfall is plausibly attributable to (a) finite calibration-set size (3,450 scores/step, pooled across only 3 folds), (b) time-series exchangeability being an approximation, not a guarantee, for pooled-across-fold financial return data, and (c) for the production-checkpoint table specifically, the extra assumption that a calibration fit on walk-forward fold models transfers to a separately-trained production model. Treat the calibrated band as **meaningfully better and much closer to trustworthy than the raw MC-Dropout band, but still somewhat overconfident** — not a fully solved calibration guarantee.

Calibration artifacts: `lstm/models/seq2seq/conformal_none.json`, `conformal_concat.json` (reusable at inference without rerunning the pipeline). Full report: `lstm/conformal_calibration_report.json` (fold boundaries, per-fold training results, full per-step coverage tables, `q_hat` arrays).

## Known limitations / open questions (stated honestly, not smoothed over)

1. Dataset size after the horizon-30 + boundary-exclusion cuts is smaller than the already-small 1-day dataset in relative terms (each window "costs" more history); watch train/val loss divergence closely as an overfitting gate, same discipline as PriceLSTM's R1 gate check.
2. Teacher-forcing-decay-vs-early-stopping interaction (see above) is a judgment call, not empirically tuned yet. In practice both variants' best checkpoint was found early (epoch 4-7, under heavy teacher forcing) — val loss is always evaluated fully-autoregressively regardless of the training-time ratio, so this isn't a leakage/inflation concern, but it does mean the "improvement" the model finds under heavy teacher forcing didn't grow much stronger as training shifted toward autoregressive — worth watching if this rework continues.
3. **MC-Dropout's raw uncertainty is badly under-calibrated** (~4-5% empirical vs. 90% nominal coverage, both variants, across the full horizon) — see "Results" above. **Fixed, but not perfectly, by conformal calibration** (see "Conformal calibration (Direction 4)" above): honest held-out coverage improves to ~83.1-83.5% against a 90% nominal target — a large improvement, not a complete one. Use the conformal-calibrated band (`lstm/models/seq2seq/conformal_{variant}.json`), not the raw MC-Dropout percentiles, for anything presented as a probability; even then, expect mild residual overconfidence rather than exact nominal coverage.
4. The flux trajectory projection's lower-bound bias (events between `as_of_date` and the target date are unknowable) is fed to the decoder as a raw scalar with no explicit bias correction — the network is left to learn around it implicitly. Whether to expose additional signal about this bias (e.g. a trend/slope feature) is an open modeling question, not resolved here.
5. `film`'s FiLM-gating hyperparameters (`film_hidden=8`) are an unvalidated default, not tuned — appropriate given `film` isn't trained by default yet.
6. Backtest Sharpe figures above are single-run, small-sample point estimates (same caveat `backtest/metrics.py` states for PriceLSTM) — directional signal only, not a claim of statistical reliability.
