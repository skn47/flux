"""
Dataset loading, vocabulary construction (from scratch, no pretrained
tokenizer/embeddings), and train/val splitting for the Phase C event-type
classifier.

Data flow, no-leakage discipline (see nlp_classifier/README.md for the full
writeup):

1. `human_reviewed_holdout.json` (hand-labeled by the NLP engineer,
   independently of the rule labeler) is the held-out evaluation set. Every
   `event_id` listed there is excluded entirely from what this module loads
   for training/val -- not just relabeled, removed.
2. The remaining `raw_events` (source='rss') JOIN `classified_events`
   (label_source='rule_based') rows are the only training signal. These
   labels are the Phase B bootstrap labeler's noisy output, not ground
   truth (see labeling/README.md's spot-check: ~72% correct on a hand
   sample). `evaluate.py` never scores against these -- only against the
   hand-reviewed holdout.
3. `gdelt`-sourced rows are excluded from this text classifier entirely:
   GDELT rows in this project's `raw_events` table carry no title/text
   (confirmed by direct query -- GDELT is an actor-interaction event feed,
   not an article corpus), so there is nothing for a *text* classifier to
   read. This is a real, documented limitation of the source (already
   flagged in progress/event_taxonomy.md §3), not an oversight here.
4. A vocabulary is built from the training split's tokens only (never from
   val or holdout text) so the "from scratch, no pretrained embeddings"
   constraint is genuinely honored end to end -- the model has literally
   never seen holdout tokens' embeddings initialized from anything but
   random init, and out-of-vocabulary words at eval time correctly map to
   <unk>.
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.labeling.schema import EVENT_TYPES

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "events.db"
HOLDOUT_PATH = REPO_ROOT / "nlp_classifier" / "human_reviewed_holdout.json"

CLASSES: list[str] = list(EVENT_TYPES)  # derived from EVENT_TYPES, order fixed for label
# indices -- count intentionally not hardcoded here (was 8 pre-S3, now 9 after S3 added
# `regulatory_approval_decision`; see labeling/schema.py) so this file never goes stale
# again. Retraining/re-evaluating the classifier against the new category is explicitly
# out of scope for S3 (see progress/sector_expansion_pharma.md) -- this is a label-space
# definition only, not a retraining trigger.
LABEL2IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX2LABEL: dict[int, str] = {i: c for c, i in LABEL2IDX.items()}

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Simple from-scratch tokenizer: lowercase, alphanumeric runs. No
    pretrained tokenizer (e.g. BPE/wordpiece) is used anywhere in this
    project, per the from-scratch constraint."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Example:
    event_id: str
    text: str
    label: str
    source_of_label: str  # "rule_based" (train/val) or "human_reviewed" (holdout)


def load_holdout() -> list[Example]:
    with open(HOLDOUT_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    out = []
    for r in rows:
        text = f"{r['title'] or ''} {r.get('text_snippet') or ''}".strip()
        out.append(
            Example(
                event_id=r["event_id"],
                text=text,
                label=r["human_label"],
                source_of_label="human_reviewed",
            )
        )
    return out


def load_training_pool(
    db_path: Path | str = DEFAULT_DB_PATH,
    exclude_event_ids: Optional[set[str]] = None,
) -> list[Example]:
    """
    All rss rows with a rule_based label, EXCLUDING exclude_event_ids
    (the holdout). GDELT rows are excluded structurally (no title/text --
    see module docstring point 3), not by an explicit filter here beyond
    the `source='rss'` join condition.
    """
    exclude_event_ids = exclude_event_ids or set()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        """
        SELECT r.id AS event_id, r.title, r.text, c.event_type
        FROM raw_events r
        JOIN classified_events c ON r.id = c.event_id
        WHERE r.source = 'rss' AND c.label_source = 'rule_based'
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    out = []
    for r in rows:
        if r["event_id"] in exclude_event_ids:
            continue
        title = r["title"] or ""
        text = r["text"] or ""
        combined = f"{title} {text}".strip()
        out.append(
            Example(
                event_id=r["event_id"],
                text=combined,
                label=r["event_type"],
                source_of_label="rule_based",
            )
        )
    return out


def build_vocab(
    examples: list[Example], min_freq: int = 1, max_size: Optional[int] = 20000
) -> dict[str, int]:
    """Vocabulary built strictly from the given examples' tokens (the
    training split only -- callers must not pass val/holdout examples here).
    """
    counts: Counter[str] = Counter()
    for ex in examples:
        counts.update(tokenize(ex.text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    ranked = [w for w, c in counts.most_common() if c >= min_freq]
    if max_size is not None:
        ranked = ranked[: max_size - len(vocab)]
    for w in ranked:
        vocab[w] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    tokens = tokenize(text)[:max_len]
    ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in tokens]
    if len(ids) < max_len:
        ids = ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids


def stratified_split(
    examples: list[Example], val_frac: float = 0.15, seed: int = 42
) -> tuple[list[Example], list[Example]]:
    """
    Splits per-class so every class with >=2 examples contributes to both
    splits where possible; classes with exactly 1 example go entirely to
    train (a single example cannot be usefully split, and a val example
    with no corresponding train signal is not a meaningful test of
    anything -- see nlp_classifier/README.md for the honest accounting of
    which classes end up with zero val support here).
    """
    rng = random.Random(seed)
    by_label: dict[str, list[Example]] = {}
    for ex in examples:
        by_label.setdefault(ex.label, []).append(ex)

    train: list[Example] = []
    val: list[Example] = []
    for label, exs in by_label.items():
        exs = exs[:]
        rng.shuffle(exs)
        n_val = int(round(len(exs) * val_frac))
        # Never fully empty train for a class that has >=1 example, and only
        # send examples to val if there are enough left for train to be
        # non-trivial (>=2 total needed to split at all).
        if len(exs) < 2:
            n_val = 0
        n_val = min(n_val, len(exs) - 1) if len(exs) >= 2 else 0
        val.extend(exs[:n_val])
        train.extend(exs[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def class_support(examples: list[Example]) -> dict[str, int]:
    counts = Counter(ex.label for ex in examples)
    return {c: counts.get(c, 0) for c in CLASSES}
