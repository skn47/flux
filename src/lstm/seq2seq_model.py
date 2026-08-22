"""
Seq2seq (encoder-decoder) LSTM for the 30-day distributional price-reaction
forecast. Runs ALONGSIDE lstm/model.py::PriceLSTM (untouched, still the
1-day-ahead production model) rather than replacing it -- see
lstm/seq2seq_README.md for the full rationale and the honest side-by-side
variant comparison this feeds into (same culture as lstm/README.md's
existing baseline-vs-flux reporting).

Architecture
------------
Encoder mirrors PriceLSTM's stack for direct comparability:
    Input (batch, LOOKBACK, n_features)
    -> LSTM(hidden1=64, full sequence) -> Dropout(0.2)
    -> LSTM(hidden2=32, final hidden state only)
    -> concat with ticker Embedding(dim=4) -> Linear -> decoder h0
    (decoder c0 is a separate small Linear projection of the encoder's final
    cell state -- ticker identity does NOT flow into c0, only h0, same
    "ticker identity injected once, not every step" rationale as PriceLSTM)

Decoder is an explicit per-step unroll (nn.LSTMCell, not nn.LSTM) over
`horizon` steps, because two things a plain nn.LSTM cannot do are required
every step:
  1. autoregressive feedback -- each step's input includes the model's own
     previous prediction (teacher-forced or self-fed, see
     lstm/train_seq2seq.py's scheduled sampling)
  2. flux conditioning -- each step is conditioned on that day's PROJECTED
     flux trajectory value (flux_engine/projection.py), which varies per
     step

`flux_conditioning` modes:
  "none"   -- flux is not used at all (analogous to PriceLSTM's baseline
              variant)
  "concat" -- the flux value is concatenated into the decoder step's input
              (analogous to PriceLSTM's flux variant: flux as one more
              input column), just applied per-step instead of per-window
  "film"   -- the flux value instead generates a (gamma, beta) pair via
              FluxFiLMGenerator, which affine-modulates the decoder's raw
              hidden state (h_mod = gamma*h_raw + beta) BEFORE it is carried
              forward to the next step -- so flux has compounding
              architectural leverage on future dynamics, not just the
              current step's output. This is the concrete difference from
              "concat": there flux is one diluted feature; here it gates the
              recurrent state itself, the concrete mechanism behind this
              rework's "flux as primary alpha driver" direction (see
              lstm/seq2seq_README.md).

MC-Dropout uncertainty sampling (`sample_trajectories`, in
lstm/train_seq2seq.py) leaves this model in `.train()` mode at inference so
its Dropout layers stay active -- the same technique Nexus's TensorFlow LSTM
uses (correctly). The key difference: here each of the N sampled 30-day
paths is autoregressive end-to-end (day k's dropout-perturbed prediction
feeds day k+1's input), so the sampled paths are genuinely correlated
day-to-day, not independent per-day draws the way a non-autoregressive
Dense(30) head with per-day-independent dropout would produce.

PyTorch is used purely as a tensor/autograd library, matching lstm/model.py
and the rest of this project -- no pretrained weights.
"""

from __future__ import annotations

import torch
from torch import nn

_VALID_FLUX_CONDITIONING = ("none", "concat", "film")


class FluxFiLMGenerator(nn.Module):
    """Maps a scalar (per-step, per-window) flux value to a (gamma, beta)
    affine-modulation pair for the decoder's hidden state. A small 2-layer
    MLP rather than a single Linear -- a purely linear map from one scalar
    could only ever scale gamma/beta along one fixed direction in
    hidden-state space, which is too constrained to be a meaningful gate."""

    def __init__(self, hidden_size: int, film_hidden: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(1, film_hidden),
            nn.Tanh(),
            nn.Linear(film_hidden, 2 * hidden_size),
        )

    def forward(self, flux_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # flux_value: (batch, 1) -> gamma, beta: each (batch, hidden_size)
        out = self.net(flux_value)
        gamma, beta = out.split(self.hidden_size, dim=1)
        # gamma centered at 1.0 (not 0.0): at initialization (net's output
        # near 0 for a freshly-initialized Linear) FiLM starts close to an
        # identity transform (h_mod ~= h_raw), rather than zeroing the
        # hidden state out -- standard FiLM-init practice, avoids
        # destabilizing early training before the gate has learned anything.
        return 1.0 + gamma, beta


class Seq2SeqPriceLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_tickers: int,
        horizon: int = 30,
        flux_conditioning: str = "concat",
        ticker_embed_dim: int = 4,
        hidden1: int = 64,
        hidden2: int = 32,
        decoder_hidden: int = 32,
        film_hidden: int = 8,
        dropout: float = 0.2,
        decoder_dropout: float = 0.2,
    ):
        super().__init__()
        if flux_conditioning not in _VALID_FLUX_CONDITIONING:
            raise ValueError(
                f"Unknown flux_conditioning={flux_conditioning!r}, expected one of {_VALID_FLUX_CONDITIONING}"
            )
        self.horizon = horizon
        self.flux_conditioning = flux_conditioning
        self.decoder_hidden = decoder_hidden

        # --- Encoder (mirrors lstm/model.py::PriceLSTM) ---
        self.embedding = nn.Embedding(n_tickers, ticker_embed_dim)
        self.enc_lstm1 = nn.LSTM(input_size=n_features, hidden_size=hidden1, batch_first=True)
        self.enc_dropout1 = nn.Dropout(dropout)
        self.enc_lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)

        # Ticker identity injected once, into the decoder's initial hidden
        # state -- not re-fed every step (same rationale as PriceLSTM: keeps
        # the decoder's recurrence about price/flux dynamics only; ticker
        # identity only adjusts the starting point).
        self.h0_proj = nn.Linear(hidden2 + ticker_embed_dim, decoder_hidden)
        self.c0_proj = nn.Linear(hidden2, decoder_hidden)

        # --- Decoder ---
        decoder_input_size = 1  # previous predicted return, always present
        if flux_conditioning == "concat":
            decoder_input_size += 1  # + this step's projected flux value
        self.decoder_cell = nn.LSTMCell(input_size=decoder_input_size, hidden_size=decoder_hidden)
        self.decoder_dropout = nn.Dropout(decoder_dropout)
        self.head = nn.Linear(decoder_hidden, 1)

        self.film: FluxFiLMGenerator | None = None
        if flux_conditioning == "film":
            self.film = FluxFiLMGenerator(decoder_hidden, film_hidden=film_hidden)

    def encode(self, x: torch.Tensor, ticker_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, LOOKBACK, n_features), ticker_ids: (batch,).
        Returns decoder initial (h0, c0), each (batch, decoder_hidden)."""
        out, _ = self.enc_lstm1(x)              # (batch, LOOKBACK, hidden1)
        out = self.enc_dropout1(out)
        _, (h_n, c_n) = self.enc_lstm2(out)       # h_n, c_n: (1, batch, hidden2)
        last_hidden = h_n[-1]                     # (batch, hidden2)
        last_cell = c_n[-1]                       # (batch, hidden2)
        emb = self.embedding(ticker_ids)           # (batch, ticker_embed_dim)
        h0 = self.h0_proj(torch.cat([last_hidden, emb], dim=1))
        c0 = self.c0_proj(last_cell)
        return h0, c0

    def forward(
        self,
        x: torch.Tensor,
        ticker_ids: torch.Tensor,
        flux_traj: torch.Tensor | None = None,
        teacher_forcing: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        x: (batch, LOOKBACK, n_features) -- feature index 0 must be
            daily_return, per lstm/dataset.py::TARGET_COL's fixed position
            (BASELINE_FEATURE_COLS[0] / FLUX_FEATURE_COLS[0]).
        ticker_ids: (batch,)
        flux_traj: (batch, horizon), unscaled projected flux values (see
            flux_engine/projection.py) -- required unless
            flux_conditioning == "none".
        teacher_forcing: (batch, horizon), the TRUE scaled daily_return at
            each decode step. None (default) => fully autoregressive
            (used at inference / MC-Dropout sampling). Given, each step
            independently and randomly chooses (per example, per step, per
            `teacher_forcing_ratio`) between the true value and the model's
            own previous prediction -- see lstm/train_seq2seq.py's scheduled
            sampling.
        teacher_forcing_ratio: probability of using the true value at each
            step; ignored if teacher_forcing is None.

        Returns: (batch, horizon) standardized daily_return predictions.
        Gradients flow through autoregressively-fed predictions on purpose
        (not detached) -- the model is meant to learn from its own
        compounding trajectory error, the same design PriceLSTM's target
        already relies on daily_return being both input feature and target.
        """
        batch_size = x.size(0)
        device = x.device
        if self.flux_conditioning != "none" and flux_traj is None:
            raise ValueError(f"flux_conditioning={self.flux_conditioning!r} requires flux_traj")

        h, c = self.encode(x, ticker_ids)
        # First decode step's "previous return" is the last observed value
        # in the input window (feature index 0 is daily_return by
        # construction) -- continuity with the encoder's own history, not an
        # arbitrary zero.
        prev_return = x[:, -1, 0].unsqueeze(1)  # (batch, 1)

        outputs = []
        for k in range(self.horizon):
            if self.flux_conditioning == "concat":
                flux_k = flux_traj[:, k].unsqueeze(1)
                step_input = torch.cat([prev_return, flux_k], dim=1)
            else:
                step_input = prev_return

            h_raw, c = self.decoder_cell(step_input, (h, c))

            if self.flux_conditioning == "film":
                flux_k = flux_traj[:, k].unsqueeze(1)
                gamma, beta = self.film(flux_k)
                h = gamma * h_raw + beta
            else:
                h = h_raw

            h_for_output = self.decoder_dropout(h)
            pred_k = self.head(h_for_output)  # (batch, 1)
            outputs.append(pred_k)

            if teacher_forcing is not None:
                true_k = teacher_forcing[:, k].unsqueeze(1)
                use_true = torch.rand(batch_size, 1, device=device) < teacher_forcing_ratio
                prev_return = torch.where(use_true, true_k, pred_k)
            else:
                prev_return = pred_k

        return torch.cat(outputs, dim=1)  # (batch, horizon)
