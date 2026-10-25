import torch
import torch.nn as nn
from typing import Optional, Tuple


class ForecastDecoder(nn.Module):

    def __init__(self, context_dim: int = 32, forecast_steps: int = 4, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_steps)
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class CascadedHydrologicalDecoder(nn.Module):
    """
    Physics-guided cascaded hydrological propagation:
    SPEI-3 (surface/soil moisture) -> SPEI-6 (streamflow/hydrology) -> SPEI-12 (groundwater/aquifer).
    Includes decoupled dropout regularization (p=0.35) on the SPEI-3 head to prevent validation divergence.
    """
    def __init__(
        self,
        context_dim: int = 32,
        forecast_steps: int = 4,
        hidden_dim: int = 64,
        spei3_dropout: float = 0.35,
        base_dropout: float = 0.2
    ):
        super().__init__()
        self.forecast_steps = forecast_steps

        # Head 1: SPEI-3 (High-frequency Agricultural Head) with decoupled p=0.35
        self.spei3_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(spei3_dropout),
            nn.Linear(hidden_dim, forecast_steps)
        )

        # Head 2: SPEI-6 (Hydrological Head) conditioned on SPEI-3 forecast
        self.spei6_head = nn.Sequential(
            nn.Linear(context_dim + forecast_steps, hidden_dim),
            nn.ReLU(),
            nn.Dropout(base_dropout),
            nn.Linear(hidden_dim, forecast_steps)
        )

        # Head 3: SPEI-12 (Groundwater Head) conditioned on SPEI-6 forecast
        self.spei12_head = nn.Sequential(
            nn.Linear(context_dim + forecast_steps, hidden_dim),
            nn.ReLU(),
            nn.Dropout(base_dropout),
            nn.Linear(hidden_dim, forecast_steps)
        )

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # context: [Batch, context_dim] or [Batch, 3, context_dim]
        c3 = context[:, 0] if context.dim() == 3 else context
        c6 = context[:, 1] if context.dim() == 3 else context
        c12 = context[:, 2] if context.dim() == 3 else context

        p3 = self.spei3_head(c3)
        # Physics-guided cascade: detaching p3 to maintain decoupled gradient dynamics
        p6_in = torch.cat([c6, p3.detach()], dim=-1)
        p6 = self.spei6_head(p6_in)

        # Physics-guided cascade: detaching p6
        p12_in = torch.cat([c12, p6.detach()], dim=-1)
        p12 = self.spei12_head(p12_in)

        return p3, p6, p12


class MultiHorizonMultiHeadDecoder(nn.Module):
    """
    Phase 5 & 6 Decoder: Separate heads for SPEI-3, SPEI-6, SPEI-12.
    Supports either parallel or cascaded propagation.
    """
    def __init__(
        self,
        context_dim: int = 32,
        forecast_steps: int = 4,
        hidden_dim: int = 64,
        spei3_dropout: float = 0.35,
        base_dropout: float = 0.2,
        use_cascading: bool = True
    ):
        super().__init__()
        self.use_cascading = use_cascading
        self.forecast_steps = forecast_steps

        if use_cascading:
            self.cascade = CascadedHydrologicalDecoder(
                context_dim, forecast_steps, hidden_dim, spei3_dropout, base_dropout
            )
        else:
            self.spei3_head = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(spei3_dropout),
                nn.Linear(hidden_dim, forecast_steps)
            )
            self.spei6_head = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(base_dropout),
                nn.Linear(hidden_dim, forecast_steps)
            )
            self.spei12_head = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(base_dropout),
                nn.Linear(hidden_dim, forecast_steps)
            )

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_cascading:
            return self.cascade(context)
        c3 = context[:, 0] if context.dim() == 3 else context
        c6 = context[:, 1] if context.dim() == 3 else context
        c12 = context[:, 2] if context.dim() == 3 else context
        return self.spei3_head(c3), self.spei6_head(c6), self.spei12_head(c12)


# Standard aliases for conference & modular consistency
MultiHorizonDecoder = MultiHorizonMultiHeadDecoder
MultiTaskDecoder = MultiHorizonMultiHeadDecoder
Decoder = ForecastDecoder

