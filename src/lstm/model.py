"""
LSTM price-reaction model definition.

Architecture (documented choices, not silent defaults -- see lstm/README.md
"Architecture and hyperparameter choices" for the full reasoning). UNCHANGED
by the R1 target/feature rework (see lstm/dataset.py's module docstring) --
only n_features and the output's documented meaning changed:

    Input: (batch, 60, n_features)          n_features = 4 (baseline) or 5 (flux)
    LSTM #1: hidden_size=64, return_sequences equivalent (full output kept)
    Dropout: 0.2, applied to LSTM #1's output only (matches spec_sheet.md
             Paper 1's placement -- single dropout, after the first LSTM
             layer, not the second)
    LSTM #2: hidden_size=32 (final hidden state only, no return_sequences)
    Ticker embedding: dim=4, concatenated to LSTM #2's final hidden state
             -- see "Per-ticker or shared model" in the README for why a
             single shared model + ticker embedding was chosen over 5
             separate per-ticker models, and why the embedding is
             concatenated at the head rather than fed into the recurrent
             core at every timestep (keeps the LSTM's recurrent dynamics
             about price behavior only; ticker identity only adjusts the
             final mapping).
    Dense (output): 1 unit, linear activation, predicts the STANDARDIZED
             (z-scored, per lstm/scaling.py::StandardScaler) next-day
             `daily_return` -- NOT a scaled price level (R1 rework; the old
             design predicted the scaled `adj_close` price level, which was
             the root cause of the prior backtest's extrapolation-bias
             failure -- see lstm/dataset.py's module docstring and
             lstm/README.md's "Gate check result").

PyTorch is used purely as a tensor/autograd library, matching this
project's other torch-based package (nlp_classifier/model.py) -- no
pretrained weights, every parameter trained from scratch on this project's
own data.
"""

from __future__ import annotations

import torch
from torch import nn


class PriceLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_tickers: int,
        ticker_embed_dim: int = 4,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_tickers, ticker_embed_dim)
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.head = nn.Linear(hidden2 + ticker_embed_dim, 1)

    def forward(self, x: torch.Tensor, ticker_ids: torch.Tensor) -> torch.Tensor:
        # x: (batch, 60, n_features), ticker_ids: (batch,)
        out, _ = self.lstm1(x)              # (batch, 60, hidden1)
        out = self.dropout1(out)
        _, (h_n, _) = self.lstm2(out)        # h_n: (1, batch, hidden2)
        last_hidden = h_n[-1]                # (batch, hidden2)
        emb = self.embedding(ticker_ids)      # (batch, ticker_embed_dim)
        combined = torch.cat([last_hidden, emb], dim=1)
        return self.head(combined).squeeze(-1)  # (batch,) standardized daily_return prediction
