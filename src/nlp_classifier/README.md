# NLP Classifier (Phase C) — Proof-of-Mechanics Training Pipeline

This package trains and evaluates a from-scratch neural **event-type classifier** on top of
Phase A's `raw_events` and Phase B's `classified_events` (`data/events.db`). Per an explicit,
already-made scope decision (see below), this is built and reported **as a proof that the
training/eval mechanics work correctly** — data loading, no-leakage splitting, a genuinely
from-scratch train loop, and honest evaluation — **not** as an attempt to ship a production-
quality classifier on ~100 noisy, imbalanced examples. Results are reported as observed,
including where the model does not clearly beat a trivial baseline.

## Scope call: event-type classification only, NER deferred

The original Phase C plan called for both an event-type classifier and an entity/country/sector
extractor. This implementation **scopes the trained model to event-type classification only**:

- Country and company/ticker extraction already works deterministically via keyword matching in
  `labeling/rule_labeler.py` (reusing `config/mvp_scope.py`) and is kept as the interim entity
  solution — it is not retrained or replaced here.
- There is currently **zero span-level (token/entity-boundary) annotated data** anywhere in this
  project. Phase B (`labeling/schema.py`, `progress/event_taxonomy.md`) only ever produced
  *document-level* tags (`countries`/`sector`/`companies` as whole-document lists), never
  annotated character/token spans within an article. A real extractive NER model (from-scratch,
  per this project's constraint — no pretrained NER model either) needs span-labeled training
  examples to learn boundaries from; with none in hand, there is nothing to train against.
  Fabricating spans from the existing keyword-match hits would just be re-deriving the rule
  labeler's own output as fake "ground truth," which is circular, not real signal.
- **Real NER training is explicitly deferred as future work**, contingent on span-labeled data
  being produced (e.g. a manual span-annotation pass, or LLM-assisted span extraction that is
  itself spot-checked like the Phase B bootstrap labelers were) — flagged here, not silently
  dropped from scope.

## No-leakage discipline

1. `nlp_classifier/human_reviewed_holdout.json` — 54 `raw_events` rows, each read and
   independently re-labeled by hand against `progress/event_taxonomy.md`'s category definitions
   (**not** copied from the existing `rule_based` label — see per-row `human_reasoning`). Every
   `event_id` in this file is a `event_type` ground-truth judgment call, with the corresponding
   `rule_based_label` kept alongside purely for comparison.
2. Every one of those 54 `event_id`s is excluded entirely from `nlp_classifier/dataset.py`'s
   training-pool query (`load_training_pool(exclude_event_ids=...)`) — not relabeled, structurally
   absent from what the model ever trains or is tuned (via validation) against.
3. `train.py` trains only on the remaining rows' existing noisy `rule_based` labels.
4. `evaluate.py` scores **only** against the hand-reviewed holdout. It never touches
   `rule_based`/`gdelt_derived` labels for scoring — doing so would just reward the model for
   agreeing with the rule labeler's own documented errors (see `labeling/README.md`'s spot-check:
   ~72% correct on a 25-row sample), which is circular, not a real generalization test.

### Holdout construction

Random uniform sampling over `raw_events` would draw overwhelmingly from `unclassified` /
`corporate_strategic` / `geopolitical_military_tension` (the corpus's dominant classes) and
produce near-zero signal on the five rarer categories. Instead, the 54-row holdout was built by
stratified sampling over the existing `rule_based` label groups (seed 42), deliberately
oversampling rarer classes, then every row's actual title/text was read by hand and assigned an
independent `event_type` judgment (not copied from the rule label):

| rule_based stratum | rows in that stratum (rss corpus) | taken into holdout |
|---|---|---|
| `supply_chain_fab_disruption` | 1 | 1 (all of it) |
| `natural_disaster` | 3 | 2 |
| `trade_export_control` | 4 | 3 |
| `regulatory_subsidy_action` | 4 | 3 |
| `macroeconomic_conditions` | 8 | 5 |
| `geopolitical_military_tension` | 10 | 5 |
| `corporate_strategic` | 44 | 10 |
| `unclassified` | 376 | 25 (random, to surface false negatives) |

GDELT rows were excluded from holdout candidacy — see "GDELT rows excluded" below.

Hand-review vs. rule-label **agreement was 33/54 (61%)** — lower than Phase B's own 72%
non-unclassified spot-check, expected because this sample was deliberately built to oversample
ambiguous/rare-class/false-negative cases rather than a uniform sample. Concrete rule-labeler
misses this review surfaced (documented per-row in the JSON, `human_reasoning` field), beyond the
ones already known from `labeling/README.md`:

- **"Bosch begins sample production at its first US semiconductor plant"** — a real
  `corporate_strategic` capacity milestone the rule dictionary missed because its exact-phrase
  keyword ("capacity expansion") doesn't appear in this headline.
- **"The $3.3 trillion chip sell-off is nearing a bear market"** and **"South Korea's turbulence
  seen as boon for Hong Kong stocks"** — real `macroeconomic_conditions` content missed because
  neither contains any of the rule dictionary's literal rate/inflation/GDP/Fed keywords.
- **"Chip worker shortage puts U.S. semiconductor boom on the brink"** — the same recall gap
  already flagged in `labeling/README.md` (dictionary has `"chip shortage"`, not `"chip worker
  shortage"`), reconfirmed independently here.
- Several `rule_based` non-`unclassified` calls judged wrong on inspection: a Japan frozen-food
  cyberattack story tagged `supply_chain_fab_disruption` (off-sector false positive), a Hong Kong
  audit-regulator story tagged `regulatory_subsidy_action` (not semiconductor-targeted), routine
  daily rate-table posts tagged `macroeconomic_conditions`, and several off-sector/off-tracked-
  company headlines (SpaceX, Fanatics, DeepSeek, Anthropic-Meta) tagged `corporate_strategic`.

### GDELT rows excluded (structural, not a choice made here)

`raw_events.title`/`text` are `NULL` for every `gdelt`-sourced row (confirmed by direct query) —
GDELT is an actor-interaction feed, not an article corpus (see `progress/event_taxonomy.md` §3). A
*text* classifier has nothing to read for these rows, so `dataset.py` only ever queries
`source='rss'` rows. This means the classifier is trained/evaluated purely on the 450 RSS rows
(minus the 54-row holdout = 396 available for train/val), not the full 539-row corpus.

## Files

- `dataset.py` — loads training rows (excluding the holdout), tokenizes with a from-scratch
  whitespace/alphanumeric tokenizer (no pretrained tokenizer), builds a vocabulary from the
  **training split's own tokens only**, and produces a stratified train/val split.
- `model.py` — `EventTypeClassifier`: randomly-initialized embedding (dim 64) → single-layer
  BiLSTM (hidden dim 64) → linear head over the 8 classes (`labeling.schema.EVENT_TYPES`). No
  pretrained weights anywhere (verified: `torch` used only as a tensor/autograd library, no
  `torchtext`/HF/pretrained embedding file loaded).
- `train.py` — fixed seed (42), class-weighted cross-entropy (inverse train-frequency, to avoid a
  trivial always-`unclassified` collapse), checkpoints on best **val macro-F1** (not accuracy —
  accuracy alone rewards the majority class on this imbalance), logs per-epoch train/val
  loss/accuracy/macro-F1.
- `evaluate.py` — precision/recall/F1 per class + macro-F1 against the hand-reviewed holdout,
  reports per-class support, and computes a trivial majority-class baseline (always predict the
  training pool's most common class, `unclassified`) for honest comparison.
- `human_reviewed_holdout.json` — the 54-row hand-labeled evaluation set (`event_id`, `title`,
  `text_snippet`, `rule_based_label`, `human_label`, `human_reasoning`).
- `checkpoints/` — `best_model.pt` (checkpoint), `vocab.json` (the training-split vocabulary).

## How to run

```
./.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # if not present
./.venv/bin/python -m nlp_classifier.train
./.venv/bin/python -m nlp_classifier.evaluate
```

## Real results (this run, seed 42)

### Support at every stage

| class | train | val | holdout |
|---|---|---|---|
| `trade_export_control` | 1 | 0 | 2 |
| `geopolitical_military_tension` | 4 | 1 | 2 |
| `supply_chain_fab_disruption` | **0** | 0 | 1 |
| `natural_disaster` | 1 | 0 | 2 |
| `regulatory_subsidy_action` | 1 | 0 | 2 |
| `macroeconomic_conditions` | 3 | 0 | 5 |
| `corporate_strategic` | 29 | 5 | 5 |
| `unclassified` | 298 | 53 | 35 |
| **total** | **337** | **59** | **54** |

`supply_chain_fab_disruption` has **zero training examples** — the entire corpus's single
`rule_based`-labeled `rss` row for this class was assigned to the holdout (evaluation signal was
judged more valuable than one training example, which teaches essentially nothing anyway). The
model structurally cannot learn this class; it never predicts it. Five other classes
(`trade_export_control`, `natural_disaster`, `regulatory_subsidy_action`, `macroeconomic_conditions`
at 3, `geopolitical_military_tension` at 4) have 0-4 training examples each and 0 or 1 val
examples — far too few to expect real generalization, and val metrics for the 0-val classes are
undefined (not computed / reported as N/A, not silently treated as 0).

### Training curve (peak val macro-F1: epoch 6, 0.4569 — but on only 59 val rows, 53 of them
`unclassified`, so this number is dominated by one class and not itself a rigorous signal; the
holdout numbers below are the real evaluation)

Train accuracy reaches ~99% by epoch 15 (near-total memorization of 337 examples with a
3,083-word from-scratch vocabulary and no dropout-defeating regularization strong enough to
prevent it) while val macro-F1 fluctuates and peaks early — a classic small-data overfitting
signature, exactly what's expected here, not a bug.

### Per-class precision / recall / F1 against the hand-reviewed holdout (the real test)

| class | n (holdout) | precision | recall | F1 |
|---|---|---|---|---|
| `trade_export_control` | 2 | 0.0000 | 0.0000 | 0.0000 |
| `geopolitical_military_tension` | 2 | 0.0000 | 0.0000 | 0.0000 |
| `supply_chain_fab_disruption` | 1 | 0.0000 | 0.0000 | 0.0000 |
| `natural_disaster` | 2 | 0.0000 | 0.0000 | 0.0000 |
| `regulatory_subsidy_action` | 2 | 0.0000 | 0.0000 | 0.0000 |
| `macroeconomic_conditions` | 5 | 0.0000 | 0.0000 | 0.0000 |
| `corporate_strategic` | 5 | 0.2308 | 0.6000 | 0.3333 |
| `unclassified` | 35 | 0.6585 | 0.7714 | 0.7105 |
| **macro-F1 (8 classes present)** | 54 | | | **0.1305** |
| **overall accuracy** | 54 | | | **0.5556** |

### Trivial majority-class baseline (always predict `unclassified`, the training pool's most
common class, 298/337 ≈ 88% of train)

| | accuracy | macro-F1 |
|---|---|---|
| **Model** | 0.5556 | **0.1305** |
| **Majority baseline** | **0.6481** | 0.0983 |

The model's **raw accuracy is worse than the trivial baseline** (0.5556 vs. 0.6481) — it predicts
`corporate_strategic` often enough (any time company/finance-sounding vocabulary appears) that it
loses some `unclassified` calls the baseline would have gotten for free. Its **macro-F1 is
slightly higher** (0.1305 vs. 0.0983) only because it manages nonzero recall/precision on
`corporate_strategic` (3/5 correct), which the constant-baseline gets zero credit for by
construction. **This is a marginal, not a real, win** — the model shows no signal at all on the
other 6 classes (0.0 F1 on all of them, including on `corporate_strategic`'s neighbors that had
any training data). Framed honestly: **the pipeline runs correctly end-to-end, but on this data
volume the model has not learned a generalizable event-type classifier** — it has learned "predict
`unclassified` unless finance/company vocabulary fires, in which case predict
`corporate_strategic`," which happens to be *slightly* better than always guessing the single
majority class, and no better than that anywhere else.

### Concrete example predictions

**Good:**
- `gold=corporate_strategic pred=corporate_strategic` — "Taiwan Semiconductor Q2 Earnings Beat
  Estimates, Revenues Rise Y/Y" (TSM earnings vocabulary the model saw analogues of in training)
- `gold=unclassified pred=unclassified` — "Why Is Taiwan Semiconductor Stock Falling Friday?"
  (correctly does not over-fire on the TSM/semiconductor terms alone)

**Bad:**
- `gold=natural_disaster pred=unclassified` — "Landslide buries residents in southwest China's
  Chongqing" (0 training examples for this class — the model has never seen the word "landslide"
  associated with any label but `unclassified`, so it cannot do anything else)
- `gold=unclassified pred=corporate_strategic` — "Meta Platforms Is Up 21% This Month, and Here Is
  What's Driving the Surge" (false positive: the model appears to have learned "stock/company-name
  + percentage move" as a shallow `corporate_strategic` cue, which also misfires on off-corridor
  companies like Meta, Fanatics, and DeepSeek)

## Honest read: does this validate the pipeline, or surface bugs?

**Validates the mechanics, does not validate the model as usable.** Every stage — the no-leakage
holdout exclusion, from-scratch vocabulary/tokenization, class-weighted training loop, checkpoint
selection on macro-F1 rather than accuracy, and per-class evaluation against genuinely independent
hand labels — runs correctly and produces internally consistent numbers (e.g. the model's train
accuracy climbing to ~99% while holdout performance stays near-baseline is exactly the overfitting
signature expected from ~337 training rows across 8 classes, most of which have single-digit
counts). No implementation bug was found in the modeling code itself.

The exercise did surface real **data bugs/gaps**, which is a legitimate and expected output of
this exercise: the hand-review's 61% agreement with `rule_based` labels (lower than Phase B's own
72% spot-check, because this sample deliberately targeted rare/ambiguous rows) reconfirms and adds
concrete new examples to the known rule-labeler limitations already catalogued in
`labeling/README.md` (exact-phrase recall gaps, off-sector false positives on generic keyword
hits). None of that is fixed here — it is Phase B's rule dictionary, out of this package's scope —
but it is now backed by fresh, independently-derived evidence rather than repeating the same 25-row
sample.

## Limitations (explicit, not hidden)

1. **Data volume is the binding constraint, not model architecture.** 337 training rows across 8
   classes, with 5 classes at 0-4 examples, is not enough for any from-scratch model to
   generalize — this was the expected, accepted outcome per the scope decision that authorized
   this build.
2. **`supply_chain_fab_disruption` has zero training examples** and cannot be predicted by this
   model at all under any input.
3. **`trade_export_control`, `natural_disaster`, `regulatory_subsidy_action`,
   `macroeconomic_conditions` have 0 val examples** — their val-split metrics during training are
   undefined, not silently reported as 0; only the holdout numbers (which do have support for
   these classes) are meaningful for them, and even those are 0.0 F1 given the training scarcity.
4. **Training labels are Phase B's noisy bootstrap output** (~72% correct per `labeling/README.md`,
   independently reconfirmed at 61% agreement on this harder, oversampled-for-rarity holdout
   sample) — the model is only as good as what it was trained on, on top of having too little of
   it.
5. **NER/entity-span extraction is out of scope for this package**, deferred pending span-labeled
   data (see "Scope call" above) — country/sector/company extraction remains
   `labeling/rule_labeler.py`'s deterministic keyword matching.
6. **GDELT rows are structurally excluded** from this text classifier (no title/text field
   populated) — only the 450 RSS rows (396 non-holdout) were ever available to train/evaluate on.
