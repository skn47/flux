"""
Evaluation against the hand-reviewed holdout ONLY (never against rule_based
labels -- scoring against the same noisy labels the model trained on would
be circular and would just reward agreeing with the rule labeler's known
errors, not real generalization).

Reports, per class: support (n), precision, recall, F1 -- plus macro-F1.
Also computes a trivial majority-class baseline (always predict the most
common class in the TRAINING pool) for honest comparison, per the persona
rule that a model not beating this baseline is a real, reportable outcome.

Run: ./.venv/bin/python -m nlp_classifier.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.nlp_classifier.dataset import (
    CLASSES,
    LABEL2IDX,
    encode,
    load_holdout,
    load_training_pool,
    class_support,
)
from src.nlp_classifier.model import EventTypeClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CKPT_DIR = REPO_ROOT / "nlp_classifier" / "checkpoints"
VOCAB_PATH = CKPT_DIR / "vocab.json"
CKPT_PATH = CKPT_DIR / "best_model.pt"


def load_model_and_vocab():
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = EventTypeClassifier(
        vocab_size=ckpt["vocab_size"],
        num_classes=ckpt["num_classes"],
        embed_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab, ckpt["max_len"]


def predict_batch(model, vocab, max_len, texts: list[str]) -> list[int]:
    preds = []
    with torch.no_grad():
        for text in texts:
            ids = encode(text, vocab, max_len)
            length = min(max(len(text.split()), 1), max_len)
            ids_t = torch.tensor([ids], dtype=torch.long)
            len_t = torch.tensor([length], dtype=torch.long)
            logits = model(ids_t, len_t)
            preds.append(int(logits.argmax(dim=1).item()))
    return preds


def per_class_prf(preds: list[int], golds: list[int]) -> dict[str, dict]:
    results = {}
    for c_idx, c_name in enumerate(CLASSES):
        tp = sum(1 for p, g in zip(preds, golds) if p == c_idx and g == c_idx)
        fp = sum(1 for p, g in zip(preds, golds) if p == c_idx and g != c_idx)
        fn = sum(1 for p, g in zip(preds, golds) if p != c_idx and g == c_idx)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[c_name] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return results


def macro_f1_over_supported(prf: dict[str, dict]) -> float:
    """Macro-F1 averaged only over classes with holdout support > 0 (classes
    absent from the holdout have an undefined F1, not a 0 -- including them
    as 0 would understate performance for a reason that has nothing to do
    with model quality)."""
    supported = [v["f1"] for v in prf.values() if v["support"] > 0]
    return sum(supported) / len(supported) if supported else 0.0


def main():
    holdout = load_holdout()
    holdout_support = class_support(holdout)
    print("Holdout per-class support:")
    for c in CLASSES:
        print(f"  {c:32s} n={holdout_support[c]}")
    print(f"  TOTAL = {len(holdout)}\n")

    model, vocab, max_len = load_model_and_vocab()
    texts = [ex.text for ex in holdout]
    golds = [LABEL2IDX[ex.label] for ex in holdout]
    preds = predict_batch(model, vocab, max_len, texts)

    prf = per_class_prf(preds, golds)
    print(f"{'class':32s} {'n':>4s} {'precision':>10s} {'recall':>8s} {'f1':>6s}")
    for c in CLASSES:
        r = prf[c]
        print(f"{c:32s} {r['support']:4d} {r['precision']:10.4f} {r['recall']:8.4f} {r['f1']:6.4f}")
    macro = macro_f1_over_supported(prf)
    overall_acc = sum(1 for p, g in zip(preds, golds) if p == g) / len(golds)
    print(f"\nOverall accuracy on holdout: {overall_acc:.4f}")
    print(f"Macro-F1 (over the {sum(1 for v in prf.values() if v['support']>0)} classes present in holdout): {macro:.4f}")

    # --- Trivial majority-class baseline -----------------------------------
    pool = load_training_pool(exclude_event_ids={ex.event_id for ex in holdout})
    pool_support = class_support(pool)
    majority_class = max(pool_support, key=pool_support.get)
    majority_idx = LABEL2IDX[majority_class]
    baseline_preds = [majority_idx] * len(golds)
    baseline_prf = per_class_prf(baseline_preds, golds)
    baseline_acc = sum(1 for p, g in zip(baseline_preds, golds) if p == g) / len(golds)
    baseline_macro = macro_f1_over_supported(baseline_prf)

    print(f"\n--- Trivial majority-class baseline (always predict '{majority_class}', "
          f"the training pool's most common class, n={pool_support[majority_class]}/{len(pool)}) ---")
    print(f"Baseline accuracy on holdout: {baseline_acc:.4f}")
    print(f"Baseline macro-F1 on holdout: {baseline_macro:.4f}")

    print(f"\n--- Comparison ---")
    print(f"Model    accuracy={overall_acc:.4f}  macro-F1={macro:.4f}")
    print(f"Baseline accuracy={baseline_acc:.4f}  macro-F1={baseline_macro:.4f}")
    if macro > baseline_macro:
        print("Model beats the majority-class baseline on macro-F1.")
    elif macro == baseline_macro:
        print("Model TIES the majority-class baseline on macro-F1 -- no real signal learned.")
    else:
        print("Model does NOT beat the majority-class baseline on macro-F1 -- "
              "a real, reportable outcome given the data volume, not hidden.")

    # A few concrete example predictions for the report.
    print("\n--- Example predictions (first 10 holdout rows) ---")
    for ex, p in list(zip(holdout, preds))[:10]:
        pred_label = CLASSES[p]
        mark = "OK " if pred_label == ex.label else "ERR"
        print(f"[{mark}] gold={ex.label:28s} pred={pred_label:28s} | {ex.text[:90]}")

    return {
        "holdout_support": holdout_support,
        "per_class": prf,
        "macro_f1": macro,
        "accuracy": overall_acc,
        "baseline_majority_class": majority_class,
        "baseline_accuracy": baseline_acc,
        "baseline_macro_f1": baseline_macro,
    }


if __name__ == "__main__":
    main()
