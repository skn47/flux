"""
Training loop for the from-scratch event-type classifier.

Fixed random seed for reproducibility. Prints per-epoch train/val loss and
accuracy. Checkpoints the model with the best val macro-F1 (not raw
accuracy, since the corpus is heavily class-imbalanced -- optimizing/
selecting on accuracy alone would just reward always predicting
`unclassified`) to `nlp_classifier/checkpoints/best_model.pt`.

Run: ./.venv/bin/python -m nlp_classifier.train
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.nlp_classifier.dataset import (
    CLASSES,
    LABEL2IDX,
    Example,
    build_vocab,
    class_support,
    encode,
    load_holdout,
    load_training_pool,
    stratified_split,
)
from src.nlp_classifier.model import EventTypeClassifier

SEED = 42
MAX_LEN = 64
EMBED_DIM = 64
HIDDEN_DIM = 64
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-3
VAL_FRAC = 0.15

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CKPT_DIR = REPO_ROOT / "nlp_classifier" / "checkpoints"
VOCAB_PATH = CKPT_DIR / "vocab.json"
CKPT_PATH = CKPT_DIR / "best_model.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class TorchTextDataset(Dataset):
    def __init__(self, examples: list[Example], vocab: dict[str, int], max_len: int):
        self.examples = examples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        ids = encode(ex.text, self.vocab, self.max_len)
        length = min(max(len(ex.text.split()), 1), self.max_len)
        label = LABEL2IDX[ex.label]
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


def macro_f1(preds: list[int], golds: list[int], num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = sum(1 for p, g in zip(preds, golds) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, golds) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, golds) if p != c and g == c)
        support = tp + fn
        if support == 0:
            continue  # class not present in this split -- excluded from macro avg
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_golds = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for ids, lengths, labels in loader:
            ids, lengths, labels = ids.to(device), lengths.to(device), labels.to(device)
            logits = model(ids, lengths)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * ids.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_golds.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = sum(1 for p, g in zip(all_preds, all_golds) if p == g) / len(all_golds)
    mf1 = macro_f1(all_preds, all_golds, len(CLASSES))
    return avg_loss, acc, mf1


def main():
    set_seed(SEED)
    device = torch.device("cpu")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    holdout = load_holdout()
    holdout_ids = {ex.event_id for ex in holdout}
    print(f"Holdout (excluded from training entirely): {len(holdout_ids)} event_ids")

    pool = load_training_pool(exclude_event_ids=holdout_ids)
    print(f"Training pool (non-holdout rss rows w/ rule_based label): {len(pool)}")

    train_examples, val_examples = stratified_split(pool, val_frac=VAL_FRAC, seed=SEED)
    print(f"Train: {len(train_examples)}  Val: {len(val_examples)}")

    train_support = class_support(train_examples)
    val_support = class_support(val_examples)
    holdout_support = class_support(holdout)
    print("\nPer-class support (train / val / holdout):")
    for c in CLASSES:
        t, v, h = train_support[c], val_support[c], holdout_support[c]
        flag = ""
        if t == 0:
            flag = "  <- ZERO training examples; model cannot learn this class"
        elif v == 0:
            flag = "  <- zero val examples; val metrics for this class are undefined"
        print(f"  {c:32s} train={t:4d}  val={v:3d}  holdout={h:3d}{flag}")

    vocab = build_vocab(train_examples, min_freq=1, max_size=20000)
    print(f"\nVocab size (built from train split only): {len(vocab)}")
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab, f)

    train_ds = TorchTextDataset(train_examples, vocab, MAX_LEN)
    val_ds = TorchTextDataset(val_examples, vocab, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False) if len(val_ds) else None

    model = EventTypeClassifier(
        vocab_size=len(vocab),
        num_classes=len(CLASSES),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    # Class-weighted loss: uncompensated training on this imbalance would
    # trivially collapse to always predicting `unclassified` (>85% of the
    # training pool). Weights are inverse-frequency, computed from the
    # TRAIN split only (not val/holdout, same no-leakage discipline).
    class_counts = torch.tensor(
        [max(train_support[c], 1) for c in CLASSES], dtype=torch.float
    )
    class_weights = (1.0 / class_counts)
    class_weights = class_weights / class_weights.sum() * len(CLASSES)
    print(f"\nClass weights (inverse train-frequency): "
          f"{dict(zip(CLASSES, [round(w, 2) for w in class_weights.tolist()]))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_mf1 = -1.0
    print("\nEpoch | train_loss train_acc train_mf1 | val_loss val_acc val_mf1")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_mf1 = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )
        if val_loader is not None:
            va_loss, va_acc, va_mf1 = run_epoch(
                model, val_loader, optimizer, criterion, device, train=False
            )
        else:
            va_loss, va_acc, va_mf1 = float("nan"), float("nan"), float("nan")

        print(
            f"{epoch:5d} | {tr_loss:10.4f} {tr_acc:9.4f} {tr_mf1:9.4f} | "
            f"{va_loss:8.4f} {va_acc:7.4f} {va_mf1:7.4f}"
        )

        if val_loader is not None and va_mf1 > best_val_mf1:
            best_val_mf1 = va_mf1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab_size": len(vocab),
                    "num_classes": len(CLASSES),
                    "embed_dim": EMBED_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "max_len": MAX_LEN,
                    "epoch": epoch,
                    "val_macro_f1": va_mf1,
                },
                CKPT_PATH,
            )

    if val_loader is None:
        # Degenerate case (shouldn't happen with VAL_FRAC>0 and pool>1) --
        # save the final-epoch model so evaluate.py has something to load.
        torch.save(
            {
                "model_state": model.state_dict(),
                "vocab_size": len(vocab),
                "num_classes": len(CLASSES),
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "max_len": MAX_LEN,
                "epoch": EPOCHS,
                "val_macro_f1": None,
            },
            CKPT_PATH,
        )

    print(f"\nBest val macro-F1: {best_val_mf1:.4f}  (checkpoint: {CKPT_PATH})")


if __name__ == "__main__":
    main()
