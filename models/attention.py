"""Temporal attention over encoder hidden states.

Canonical form (shared / head_specific): additive attention
    score_t = v^T tanh(W h_t),  alpha = softmax(scores over T),
    c = sum_t alpha_t h_t.
Ablation mode `none` is handled by the forecaster (last hidden state).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TemporalAttention(nn.Module):
    """Additive temporal attention: score_t = v^T tanh(W h_t)."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, encoder_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # encoder_outputs: [B, T, H]
        scores = self.v(torch.tanh(self.W(encoder_outputs)))  # [B, T, 1]
        weights = F.softmax(scores, dim=1)  # [B, T, 1], sum_t = 1
        context = (weights * encoder_outputs).sum(dim=1)  # [B, H]
        return context, weights


# Deprecated alias kept for older imports / checkpoints.
Attention = TemporalAttention


class MultiHorizonTemporalAttention(nn.Module):
    """Deprecated multi-query attention; prefer TemporalAttention with head_specific mode."""

    def __init__(self, hidden_dim: int = 32, num_horizons: int = 3):
        super().__init__()
        self.attentions = nn.ModuleList(
            [TemporalAttention(hidden_dim) for _ in range(num_horizons)]
        )

    def forward(self, h_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        contexts, weights = [], []
        for attn in self.attentions:
            c, a = attn(h_seq)
            contexts.append(c)
            weights.append(a.squeeze(-1))
        return torch.stack(contexts, dim=1), torch.stack(weights, dim=1)
