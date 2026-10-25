#In hydrology, drought propagates sequentially: Soil Moisture Deficit (SPEI-3) $\to$ Streamflow/Reservoir Deficit (SPEI-6) $\to$ Groundwater Depletion (SPEI-12). Instead of predicting them independently, condition the SPEI-6 head on SPEI-3's latent state, and SPEI-12 on SPEI-6's latent state, while outputting Quantiles equivalent to 0.1,0.5,0.9 for uncertainty estimation: 
import torch
import torch.nn as nn

class CascadedHydrologicalDecoder(nn.Module):
    def __init__(self, hidden_dim, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.num_q = len(quantiles)
        
        # Head 1: Agricultural Drought (SPEI-3) - Higher dropout to prevent validation divergence
        self.spei3_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.35)
        )
        self.spei3_out = nn.Linear(hidden_dim // 2, self.num_q)
        
        # Head 2: Hydrological Drought (SPEI-6) conditioned on SPEI-3 latent state
        self.spei6_latent = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.20)
        )
        self.spei6_out = nn.Linear(hidden_dim // 2, self.num_q)
        
        # Head 3: Groundwater Drought (SPEI-12) conditioned on SPEI-6 latent state
        self.spei12_latent = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.spei12_out = nn.Linear(hidden_dim // 2, self.num_q)

    def forward(self, context):
        z3 = self.spei3_latent(context[:, 0, :])
        out3 = self.spei3_out(z3)
        z6 = self.spei6_latent(torch.cat([context[:, 1, :], z3], dim=-1))
        out6 = self.spei6_out(z6)
        
        z12 = self.spei12_latent(torch.cat([context[:, 2, :], z6], dim=-1))
        out12 = self.spei12_out(z12)
        return torch.stack([out3, out6, out12], dim=1)