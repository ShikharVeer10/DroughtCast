"""Shared pyramidal LSTM encoder: LSTM(h1) -> Dropout -> LSTM(h2)."""

from __future__ import annotations

import torch
from torch import nn


class SharedLSTMEncoder(nn.Module):
    """Phase 2 encoder returning all hidden states [B, T, hidden2]."""

    def __init__(
        self,
        input_size: int = 12,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.lstm1(x)
        out1 = self.dropout(out1)
        out2, _ = self.lstm2(out1)
        return out2  # [B, T, hidden2]


class CapacityMatchedGRUEncoder(nn.Module):
    """Ablation: two-layer GRU 64 -> 32 (capacity-matched to the LSTM stack)."""

    def __init__(
        self,
        input_size: int = 12,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru1 = nn.GRU(input_size, hidden1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.gru2 = nn.GRU(hidden1, hidden2, batch_first=True)
        self.hidden2 = hidden2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru1(x)
        out = self.dropout(out)
        out, _ = self.gru2(out)
        return out


class CapacityMatchedTransformerEncoder(nn.Module):
    """Ablation: projected Transformer encoder with positional encoding."""

    def __init__(
        self,
        input_size: int = 12,
        hidden2: int = 32,
        dropout: float = 0.2,
        max_len: int = 64,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden2)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden2))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            hidden2, nhead=4, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.hidden2 = hidden2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x) + self.pos[:, : x.size(1)]
        return self.encoder(h)
