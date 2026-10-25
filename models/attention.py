import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TemporalAttention(nn.Module):
    """
    Phase 3 & 4: Scaled dot-product temporal attention.
    Learns softmax weights (alpha_1 ... alpha_T) over input time steps (default T=6).
    Returns context vector c in R^d (d=hidden_dim, default 32) and attention weights.
    Supports optional multi-horizon attention flag for DroughtCastModel interface compatibility.
    """
    def __init__(self, hidden_dim: Optional[int] = None, hidden_size: Optional[int] = None,
                 num_horizons: int = 3, use_multi_horizon_attn: bool = False):
        super().__init__()
        dim = hidden_dim if hidden_dim is not None else (hidden_size if hidden_size is not None else 32)
        self.hidden_dim = dim
        self.hidden_size = dim
        self.num_horizons = num_horizons
        self.use_multi_horizon_attn = use_multi_horizon_attn

        if use_multi_horizon_attn:
            self.horizon_queries = nn.Parameter(torch.randn(num_horizons, dim) * 0.02)
            self.key_proj = nn.Linear(dim, dim)
            self.val_proj = nn.Linear(dim, dim)
            self.scale = dim ** 0.5
        else:
            self.query = nn.Linear(dim, dim, bias=False)
            self.key = nn.Linear(dim, dim, bias=False)
            self.scale = dim ** 0.5

    def forward(self, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # encoder_outputs: [Batch, Seq_len, hidden_size]
        if self.use_multi_horizon_attn:
            keys = torch.tanh(self.key_proj(encoder_outputs))
            vals = self.val_proj(encoder_outputs)
            scores = torch.einsum('bth,kh->bkt', keys, self.horizon_queries) / self.scale
            attn_weights = F.softmax(scores, dim=-1)
            context = torch.einsum('bkt,bth->bkh', attn_weights, vals)
            return context, attn_weights

        last_step = encoder_outputs[:, -1:, :]       # [Batch, 1, hidden_size]
        Q = self.query(last_step)                     # [Batch, 1, hidden_size]
        K = self.key(encoder_outputs)                 # [Batch, Seq_len, hidden_size]

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [Batch, 1, Seq_len]
        weights = torch.softmax(scores, dim=-1)                # [Batch, 1, Seq_len]

        context = torch.bmm(weights, encoder_outputs).squeeze(1)  # [Batch, hidden_size]
        return context, weights.transpose(1, 2)


class MultiHorizonTemporalAttention(nn.Module):
    """
    Multi-horizon attention with distinct query vectors for SPEI-3 (Agri),
    SPEI-6 (Hydro), and SPEI-12 (Groundwater/Long-term).
    """
    def __init__(self, hidden_dim: int = 32, num_horizons: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_horizons = num_horizons
        self.horizon_queries = nn.Parameter(torch.randn(num_horizons, hidden_dim) * 0.02)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.val_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, h_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        keys = torch.tanh(self.key_proj(h_seq))   # [B, T, H]
        vals = self.val_proj(h_seq)               # [B, T, H]

        scores = torch.einsum('bth,kh->bkt', keys, self.horizon_queries) / self.scale
        attn_weights = F.softmax(scores, dim=-1)  # [B, num_horizons, T]

        context = torch.einsum('bkt,bth->bkh', attn_weights, vals)
        return context, attn_weights


# Standard aliases
Attention = TemporalAttention
