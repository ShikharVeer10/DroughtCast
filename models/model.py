from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from models.encoder import SharedEncoder
from models.attention import TemporalAttention
from models.decoder import MultiHorizonDecoder


class DroughtCastModel(nn.Module):
    def __init__(self,input_dim: int = 12,hidden_dim: int = 64,out_hidden_dim: Optional[int] = 32,num_layers: int = 2,dropout: float = 0.2,spei3_dropout: float = 0.35,forecast_steps: int = 1,num_quantiles: int = 1,use_teleconnection_gate: bool = False,use_multi_horizon_attn: bool = False,use_cascade_decoder: bool = False,ocean_feature_indices: Optional[List[int]] = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_hidden_dim = out_hidden_dim if out_hidden_dim is not None else hidden_dim
        self.forecast_steps = forecast_steps
        self.num_quantiles = num_quantiles

        # Phase 2: Shared Encoder (LSTM(64) -> Dropout -> LSTM(32))
        self.encoder = SharedEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_hidden_dim=self.out_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_teleconnection_gate=use_teleconnection_gate,
            ocean_feature_indices=ocean_feature_indices,
        )

        # Phase 3 & 4: Temporal Attention -> Context Vector c (dim = 32)
        context_dim = self.encoder.out_dim
        self.attention = TemporalAttention(
            hidden_dim=context_dim,
            num_horizons=3,
            use_multi_horizon_attn=use_multi_horizon_attn,
        )

        # Phase 5 & 6: Multi-Horizon Forecast Heads (SPEI-3, SPEI-6, SPEI-12)
        self.decoder = MultiHorizonDecoder(
            hidden_dim=context_dim,
            forecast_steps=forecast_steps,
            num_quantiles=num_quantiles,
            dropout=dropout,
            spei3_dropout=spei3_dropout,
            use_cascade_decoder=use_cascade_decoder,
        )

        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        enc_states = self.encoder(x)                        
        context, attn_weights = self.attention(enc_states)   
        self.last_attention_weights = attn_weights.detach()
        predictions = self.decoder(context)                 
        if return_attention:
            return predictions, attn_weights
        return predictions

    def get_attention_weights(self, x: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if x is not None:
            with torch.no_grad():
                _, attn_weights = self.forward(x, return_attention=True)
            return attn_weights
        return self.last_attention_weights

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DroughtCastModel":
        model_cfg = config.get("model", config)
        return cls(
            input_dim=model_cfg.get("input_dim", 12),
            hidden_dim=model_cfg.get("hidden_dim", 64),
            out_hidden_dim=model_cfg.get("out_hidden_dim", 32),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.2),
            spei3_dropout=model_cfg.get("spei3_dropout", 0.35),
            forecast_steps=model_cfg.get("forecast_steps", 1),
            num_quantiles=model_cfg.get("num_quantiles", 1),
            use_teleconnection_gate=model_cfg.get("use_teleconnection_gate", False),
            use_multi_horizon_attn=model_cfg.get("use_multi_horizon_attn", False),
            use_cascade_decoder=model_cfg.get("use_cascade_decoder", False),
            ocean_feature_indices=model_cfg.get("ocean_feature_indices", None),
        )


class UncertaintyWeightedMultiTaskLoss(nn.Module):
    def __init__(self, num_tasks: int = 3, extreme_threshold: float = -1.5, extreme_weight: float = 1.5) -> None:
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.extreme_threshold = extreme_threshold
        self.extreme_weight = extreme_weight
        self.huber = nn.HuberLoss(reduction="none", delta=1.0)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        total_loss = torch.tensor(0.0, device=preds.device)
        task_maes: List[float] = []

        for k in range(3):
            p_k = preds[:, k]
            t_k = targets[:, k]
            valid_mask = ~torch.isnan(t_k)
            if not torch.any(valid_mask):
                task_maes.append(0.0)
                continue

            p_valid = p_k[valid_mask]
            t_valid = t_k[valid_mask]

            # Upweight severe/extreme drought deficits (SPEI <= -1.5) to prevent regression to the mean
            weights = torch.where(
                t_valid <= self.extreme_threshold,
                torch.full_like(t_valid, self.extreme_weight),
                torch.ones_like(t_valid),
            )
            loss_k = torch.mean(weights * self.huber(p_valid, t_valid))

            precision = torch.exp(-self.log_vars[k])
            total_loss = total_loss + 0.5 * precision * loss_k + 0.5 * self.log_vars[k]
            task_maes.append(torch.mean(torch.abs(p_valid - t_valid)).item())

        return total_loss, task_maes

DroughtCast = DroughtCastModel
MultiTaskAttentionLSTM = DroughtCastModel
MultiHorizonForecaster = DroughtCastModel