"""
From-scratch text classifier: randomly-initialized embedding layer -> BiLSTM
encoder -> linear classification head over the 8 event-type classes.

No pretrained weights anywhere (no pretrained embeddings, no pretrained
tokenizer, no pretrained transformer backbone) -- every parameter, including
the embedding table, is trained only on this project's own labeled corpus,
per the persona's from-scratch constraint. PyTorch is used purely as a
tensor/autograd library here, not as a source of pretrained weights.
"""

from __future__ import annotations

import torch
from torch import nn


class EventTypeClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len), lengths: (batch,)
        embedded = self.embedding(token_ids)  # (batch, seq_len, embed_dim)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers * 2, batch, hidden_dim) -- take last layer's fwd/bwd
        forward_last = h_n[-2]
        backward_last = h_n[-1]
        combined = torch.cat([forward_last, backward_last], dim=1)
        combined = self.dropout(combined)
        return self.classifier(combined)  # (batch, num_classes) logits
