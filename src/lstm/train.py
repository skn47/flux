"""
Training loop for the shared-model PriceLSTM (see lstm/model.py).

Training-loop hygiene:
  - explicit `model.train()` / `model.eval()` switching every epoch
  - `optimizer.zero_grad()` before every backward pass
  - gradient clipping (max_norm=1.0) -- a DEVIATION from both source papers
    (neither mentions gradient clipping); added here because
    `features/README.md` documents real, large single-day return swings
    (e.g. INTC -26% on 2024-08-02) that are not clipped/winsorized upstream,
    so a small dataset with these swings intact is more prone to occasional
    large gradients destabilizing training. Stated explicitly as a deviation
    per this persona's own rules, not silently added.
  - early stopping on val loss (patience configurable), matching Paper 1's
    STATED intent (Sec. VI-D, "halts training if validation loss does not
    improve for 10 consecutive epochs") -- Paper 1's own code never actually
    implements this despite the prose (spec_sheet.md flags this
    inconsistency); this module actually implements it.
  - best-val-loss checkpoint kept in memory and restored at the end (not
    just the final epoch's weights).
  - fixed random seed (torch + numpy + python) for reproducibility.
  - device management: single `device` variable threaded through model and
    every tensor; this environment is CPU-only (see requirements.txt's
    `torch==2.13.0+cpu` note), but `torch.device("cuda" if
    torch.cuda.is_available() else "cpu")` is used rather than hardcoding
    "cpu", so this code runs correctly unmodified on a GPU machine too.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.lstm.dataset import Window
from src.lstm.model import PriceLSTM

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WindowTorchDataset(Dataset):
    def __init__(self, windows: list[Window]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        return (
            torch.tensor(w.X, dtype=torch.float32),
            torch.tensor(w.ticker_id, dtype=torch.long),
            torch.tensor(w.y_scaled, dtype=torch.float32),
        )


@dataclass
class TrainConfig:
    hidden1: int = 64
    hidden2: int = 32
    ticker_embed_dim: int = 4
    dropout: float = 0.2
    batch_size: int = 32
    lr: float = 1e-3
    max_epochs: int = 150
    early_stop_patience: int = 15
    grad_clip_max_norm: float = 1.0
    seed: int = SEED


@dataclass
class TrainResult:
    model: PriceLSTM
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")


def run_epoch(model, loader, optimizer, criterion, device, train: bool, grad_clip_max_norm: float):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_seen = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X, ticker_ids, y in loader:
            X, ticker_ids, y = X.to(device), ticker_ids.to(device), y.to(device)
            preds = model(X, ticker_ids)
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
    train_windows: list[Window],
    val_windows: list[Window],
    n_features: int,
    n_tickers: int,
    config: TrainConfig = TrainConfig(),
    device: torch.device | None = None,
    verbose: bool = True,
) -> TrainResult:
    set_seed(config.seed)
    device = device or get_device()

    train_loader = DataLoader(
        WindowTorchDataset(train_windows), batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        WindowTorchDataset(val_windows), batch_size=config.batch_size, shuffle=False
    )

    model = PriceLSTM(
        n_features=n_features,
        n_tickers=n_tickers,
        ticker_embed_dim=config.ticker_embed_dim,
        hidden1=config.hidden1,
        hidden2=config.hidden2,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.MSELoss()  # matches both source papers, spec_sheet.md §4
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0
    history: list[dict] = []

    if verbose:
        print(f"{'epoch':>5s} | {'train_mse':>10s} {'val_mse':>10s}")
    for epoch in range(1, config.max_epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True,
            grad_clip_max_norm=config.grad_clip_max_norm,
        )
        val_loss = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False,
            grad_clip_max_norm=config.grad_clip_max_norm,
        )
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        if verbose:
            print(f"{epoch:5d} | {train_loss:10.6f} {val_loss:10.6f}")

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= config.early_stop_patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(no val improvement for {config.early_stop_patience} epochs; "
                        f"best epoch was {best_epoch}, best val_mse={best_val_loss:.6f})"
                    )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(model=model, history=history, best_epoch=best_epoch, best_val_loss=best_val_loss)
