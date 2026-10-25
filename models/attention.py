import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHorizonTemporalAttention(nn.Module):
    def __init__(self, hidden_dim, num_horizons=3):
        super().__init__()
        # Separate query projections for SPEI-3, SPEI-6, and SPEI-12
        self.horizon_queries = nn.Parameter(torch.randn(num_horizons, hidden_dim))
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.val_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, h_seq):
        # h_seq: [B, T, H]
        keys = torch.tanh(self.key_proj(h_seq))   # [B, T, H]
        vals = self.val_proj(h_seq)               # [B, T, H]
        
        # Compute attention scores per horizon: [B, num_horizons, T]
        scores = torch.einsum('bth,kh->bkt', keys, self.horizon_queries) / self.scale
        attn_weights = F.softmax(scores, dim=-1)  # [B, 3, T]
        
        # Context vectors per horizon: [B, 3, H]
        context = torch.einsum('bkt,bth->bkh', attn_weights, vals)
        return context, attn_weights