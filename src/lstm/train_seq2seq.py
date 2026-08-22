"""
Training loop for Seq2SeqPriceLSTM (see lstm/seq2seq_model.py).

Extends lstm/train.py's hygiene -- explicit model.train()/eval() switching,
optimizer.zero_grad() before every backward pass, gradient clipping
(max_norm=1.0, same deviation-from-source-papers rationale as lstm/train.py:
real single-day return swings in this dataset can destabilize a small
model's gradients), early stopping on val loss, best-val-loss checkpoint
kept in memory and restored at the end, fixed seed -- to the seq2seq/
multi-horizon setting, plus two genuinely new pieces this setting requires:

  1. Scheduled sampling. The decoder is autoregressive (each step's input
     includes its own previous prediction), so training purely with teacher
     forcing (always feeding the TRUE previous return) would never expose
     the model to its own compounding prediction errors during training --
     only at inference, a well-known train/inference mismatch for
     autoregressive models. `teacher_forcing_ratio` decays LINEARLY from
     `teacher_forcing_start` to `teacher_forcing_end` over
     `teacher_forcing_decay_epochs`, decided PER EXAMPLE PER STEP (not one
     coin flip per batch), so training gradually transitions toward the same
     fully-autoregressive regime used at inference. At inference / MC-Dropout
     sampling time, teacher_forcing is always None (ratio effectively 0).

  2. MC-Dropout trajectory sampling (`sample_trajectories`). Leaves the
     model in `.train()` mode (dropout active) and runs N full autoregressive
     forward passes per input window, each explicitly seeded
     (`base_seed + i`) for reproducibility -- re-running with the same
     base_seed must produce byte-identical output, or any calibration
     diagnostic built on top of it (backtest/seq_calibration.py) would
     silently differ run to run, undermining any before/after comparison.
     Because each pass is autoregressive end-to-end, the N sampled 30-day
     paths are genuinely correlated day-to-day -- unlike Nexus's TensorFlow
     LSTM, whose MC-Dropout is real but applied to 30 INDEPENDENT Dense
     outputs from one forward pass (see lstm/seq2seq_README.md for the full
     comparison).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.lstm.seq2seq_model import Seq2SeqPriceLSTM
from src.lstm.seq_dataset import HORIZON, SeqWindow

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeqWindowTorchDataset(Dataset):
    def __init__(self, windows: list[SeqWindow]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        return (
            torch.tensor(w.X, dtype=torch.float32),
            torch.tensor(w.ticker_id, dtype=torch.long),
            torch.tensor(w.y_scaled, dtype=torch.float32),   # (horizon,)
            torch.tensor(w.flux_traj, dtype=torch.float32),  # (horizon,)
        )


@dataclass
class Seq2SeqTrainConfig:
    hidden1: int = 64
    hidden2: int = 32
    ticker_embed_dim: int = 4
    decoder_hidden: int = 32
    film_hidden: int = 8
    dropout: float = 0.2
    decoder_dropout: float = 0.2
    horizon: int = HORIZON
    flux_conditioning: str = "concat"   # "none" | "concat" | "film"
    batch_size: int = 32
    lr: float = 1e-3
    max_epochs: int = 150
    early_stop_patience: int = 15
    grad_clip_max_norm: float = 1.0
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.0
    # 0 (default) means "derive as round(0.6 * max_epochs)" -- see __post_init__.
    # Explicit non-zero values are used as-is.
    teacher_forcing_decay_epochs: int = 0
    mc_dropout_n_samples: int = 50
    seed: int = SEED

    def __post_init__(self):
        if self.teacher_forcing_decay_epochs == 0:
            self.teacher_forcing_decay_epochs = max(1, round(0.6 * self.max_epochs))


@dataclass
class Seq2SeqTrainResult:
    model: Seq2SeqPriceLSTM
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")


def teacher_forcing_ratio_for_epoch(epoch: int, config: Seq2SeqTrainConfig) -> float:
    """Linear decay from teacher_forcing_start to teacher_forcing_end over
    teacher_forcing_decay_epochs, then held at teacher_forcing_end.
    `epoch` is 1-indexed, matching lstm/train.py's convention."""
    progress = min(1.0, (epoch - 1) / config.teacher_forcing_decay_epochs)
    return config.teacher_forcing_start + progress * (config.teacher_forcing_end - config.teacher_forcing_start)


def run_epoch(
    model, loader, optimizer, criterion, device, train: bool,
    grad_clip_max_norm: float, teacher_forcing_ratio: float,
):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_seen = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X, ticker_ids, y, flux_traj in loader:
            X, ticker_ids = X.to(device), ticker_ids.to(device)
            y, flux_traj = y.to(device), flux_traj.to(device)
            # Val/eval is always fully autoregressive (teacher_forcing=None)
            # -- the same regime the model is actually used under at
            # inference, so val loss is a true measure of that regime, not
            # an optimistic teacher-forced proxy.
            tf = y if train else None
            ratio = teacher_forcing_ratio if train else 0.0
            preds = model(X, ticker_ids, flux_traj=flux_traj, teacher_forcing=tf, teacher_forcing_ratio=ratio)
            loss = criterion(preds, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
                optimizer.step()
            total_loss += loss.item() * X.size(0)
            n_seen += X.size(0)
    return total_loss / n_seen


def train_model(
    train_windows: list[SeqWindow],
    val_windows: list[SeqWindow],
    n_features: int,
    n_tickers: int,
    config: Seq2SeqTrainConfig = Seq2SeqTrainConfig(),
    device: torch.device | None = None,
    verbose: bool = True,
) -> Seq2SeqTrainResult:
    set_seed(config.seed)
    device = device or get_device()

    train_loader = DataLoader(
        SeqWindowTorchDataset(train_windows), batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        SeqWindowTorchDataset(val_windows), batch_size=config.batch_size, shuffle=False
    )

    model = Seq2SeqPriceLSTM(
        n_features=n_features,
        n_tickers=n_tickers,
        horizon=config.horizon,
        flux_conditioning=config.flux_conditioning,
        ticker_embed_dim=config.ticker_embed_dim,
        hidden1=config.hidden1,
        hidden2=config.hidden2,
        decoder_hidden=config.decoder_hidden,
        film_hidden=config.film_hidden,
        dropout=config.dropout,
        decoder_dropout=config.decoder_dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0
    history: list[dict] = []

    # Early stopping is suppressed until the teacher-forcing decay schedule
    # completes -- otherwise a model that looks good under heavy teacher
    # forcing could trigger early stopping before ever training under
    # mostly-autoregressive conditions, the regime it is actually evaluated
    # under at inference. Flagged as an open tuning question in
    # lstm/seq2seq_README.md -- revisit if this proves too conservative.
    min_epochs_before_early_stop = config.teacher_forcing_decay_epochs

    if verbose:
        print(f"{'epoch':>5s} | {'tf_ratio':>8s} {'train_mse':>10s} {'val_mse':>10s}")
    for epoch in range(1, config.max_epochs + 1):
        tf_ratio = teacher_forcing_ratio_for_epoch(epoch, config)
        train_loss = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True,
            grad_clip_max_norm=config.grad_clip_max_norm, teacher_forcing_ratio=tf_ratio,
        )
        val_loss = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False,
            grad_clip_max_norm=config.grad_clip_max_norm, teacher_forcing_ratio=0.0,
        )
        history.append({"epoch": epoch, "teacher_forcing_ratio": tf_ratio,
                         "train_mse": train_loss, "val_mse": val_loss})
        if verbose:
            print(f"{epoch:5d} | {tf_ratio:8.3f} {train_loss:10.6f} {val_loss:10.6f}")

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epoch >= min_epochs_before_early_stop and epochs_since_improve >= config.early_stop_patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(no val improvement for {config.early_stop_patience} epochs, "
                        f"post-teacher-forcing-decay; best epoch was {best_epoch}, "
                        f"best val_mse={best_val_loss:.6f})"
                    )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return Seq2SeqTrainResult(model=model, history=history, best_epoch=best_epoch, best_val_loss=best_val_loss)


@torch.no_grad()
def sample_trajectories(
    model: Seq2SeqPriceLSTM,
    X: torch.Tensor,
    ticker_ids: torch.Tensor,
    flux_traj: torch.Tensor | None,
    n_samples: int = 50,
    base_seed: int = SEED,
    device: torch.device | None = None,
) -> torch.Tensor:
    """MC-Dropout trajectory sampling (see module docstring). Leaves `model`
    in `.train()` mode so its Dropout layers stay active -- the entire point
    of MC-Dropout -- and runs `n_samples` full autoregressive forward passes
    (`teacher_forcing=None` => fully self-fed, the same regime used at real
    inference), each seeded `base_seed + i`.

    Restores the model's original train/eval mode before returning, so a
    caller that happens to call this mid-training doesn't silently leave the
    model in the wrong mode for whatever runs next.

    Returns (n_samples, batch, horizon).
    """
    device = device or next(model.parameters()).device
    X, ticker_ids = X.to(device), ticker_ids.to(device)
    if flux_traj is not None:
        flux_traj = flux_traj.to(device)

    was_training = model.training
    model.train()
    try:
        samples = []
        for i in range(n_samples):
            torch.manual_seed(base_seed + i)
            preds = model(X, ticker_ids, flux_traj=flux_traj, teacher_forcing=None)
            samples.append(preds.unsqueeze(0))
        return torch.cat(samples, dim=0)  # (n_samples, batch, horizon)
    finally:
        model.train(was_training)
