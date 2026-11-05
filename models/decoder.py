from __future__ import annotations

import torch
from torch import nn


class ForecastDecoder(nn.Module):
    def __init__(
        self,
        context_dim: int = 32,
        forecast_steps: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_steps),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class CascadedHydrologicalDecoder(nn.Module):
    def __init__(
        self,
        context_dim: int = 32,
        forecast_steps: int = 4,
        hidden_dim: int = 64,
        spei3_dropout: float = 0.35,
        base_dropout: float = 0.2,
    ):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.spei3_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(spei3_dropout),
            nn.Linear(hidden_dim, forecast_steps),
        )
        self.spei6_head = nn.Sequential(
            nn.Linear(context_dim + forecast_steps, hidden_dim),
            nn.ReLU(),
            nn.Dropout(base_dropout),
            nn.Linear(hidden_dim, forecast_steps),
        )
        self.spei12_head = nn.Sequential(
            nn.Linear(context_dim + forecast_steps, hidden_dim),
            nn.ReLU(),
            nn.Dropout(base_dropout),
            nn.Linear(hidden_dim, forecast_steps),
        )

    def forward(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3 = context[:, 0] if context.dim() == 3 else context
        c6 = context[:, 1] if context.dim() == 3 else context
        c12 = context[:, 2] if context.dim() == 3 else context
        p3 = self.spei3_head(c3)
        p6 = self.spei6_head(torch.cat([c6, p3.detach()], dim=-1))
        p12 = self.spei12_head(torch.cat([c12, p6.detach()], dim=-1))
        return p3, p6, p12


# Aliases
Decoder = ForecastDecoder
MultiHorizonDecoder = CascadedHydrologicalDecoder
MultiTaskDecoder = CascadedHydrologicalDecoder
MultiHorizonMultiHeadDecoder = CascadedHydrologicalDecoder
