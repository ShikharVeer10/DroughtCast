"""Canonical DroughtCast multi-scale SPEI forecaster (diagram architecture)."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from models.attention import TemporalAttention
from models.config_schema import validate_config
from models.decoder import CascadedHydrologicalDecoder, ForecastDecoder
from models.encoder import (
    CapacityMatchedGRUEncoder,
    CapacityMatchedTransformerEncoder,
    SharedLSTMEncoder,
)


class DroughtCast(nn.Module):
    """Phase 1–6 diagram model: shared LSTM encoder, one temporal attention, three scale heads.

    Input  : [B, 6, D]
    Encoder: LSTM(64) -> Dropout(0.2) -> LSTM(32) -> [B, 6, 32]
    Attn   : additive score_t = v^T tanh(W h_t), weights [B, 6, 1]
    Context: c in R^32 shared by all heads
    Heads  : SPEI-3/6/12 accumulation-scale decoders -> T+1..T+4
    Output : predictions dict or stacked [B, 3, 4]
    """

    SCALE_NAMES = ("spei_3", "spei_6", "spei_12")

    def __init__(self, config):
        super().__init__()
        validate_config(config)
        data, model = config["data"], config["model"]
        input_dim = data.get("input_dim") or model.get("input_dim")
        if not input_dim and "features" in data:
            input_dim = len(data["features"])
            data["input_dim"] = input_dim
        if not input_dim:
            raise ValueError("data.input_dim must be set or inferable from data.features")

        self.hidden1 = int(model.get("hidden_dim_1", 64))
        self.hidden2 = int(model.get("hidden_dim_2", 32))
        dropout = float(model.get("dropout", 0.2))
        self.forecast_steps = int(data["forecast_steps"])
        encoder_type = str(model.get("encoder", "lstm")).lower()
        attention_mode = str(model.get("attention", "shared")).lower()
        heads = str(model.get("heads", model.get("head_mode", "direct"))).lower()
        self.attention_mode = attention_mode
        self.head_mode = heads
        self.quantiles = model.get("quantiles")
        self.consistency_weight = float(model.get("consistency_weight", 0.0) or 0.0)
        self.static_context = bool(model.get("static_context", False))
        self.use_cascading = bool(model.get("use_cascaded_decoder", False))

        def make_encoder():
            if encoder_type == "lstm":
                return SharedLSTMEncoder(input_dim, self.hidden1, self.hidden2, dropout)
            if encoder_type == "gru":
                return CapacityMatchedGRUEncoder(input_dim, self.hidden1, self.hidden2, dropout)
            if encoder_type == "transformer":
                return CapacityMatchedTransformerEncoder(input_dim, self.hidden2, dropout)
            raise ValueError(f"Unknown encoder: {encoder_type}")

        self.separate_encoders = bool(model.get("separate_encoders", False))
        if self.separate_encoders:
            self.encoders = nn.ModuleList([make_encoder() for _ in range(3)])
            self.encoder = None
        else:
            self.encoder = make_encoder()
            self.encoders = None

        if attention_mode == "none":
            self.attentions = None
        elif attention_mode == "shared":
            self.attentions = nn.ModuleList([TemporalAttention(self.hidden2)])
        elif attention_mode == "head_specific":
            self.attentions = nn.ModuleList([TemporalAttention(self.hidden2) for _ in range(3)])
        else:
            raise ValueError(f"Unknown attention mode: {attention_mode}")

        self.variable_gate = (
            nn.Sequential(nn.Linear(input_dim, input_dim), nn.Softmax(dim=-1))
            if model.get("variable_gate", False) else None
        )

        # Context may grow if static descriptors are concatenated (Phase 2 flag).
        context_dim = self.hidden2
        self._static_dim = 0
        if self.static_context:
            # lat + aridity (+ optional elevation placeholder slot filled by zeros if absent)
            self._static_dim = 2
            context_dim = self.hidden2 + self._static_dim

        if self.use_cascading and not self.quantiles:
            spei3_drop = model.get("spei3_dropout", 0.35)
            self.cascade_decoder = CascadedHydrologicalDecoder(
                context_dim=context_dim,
                forecast_steps=self.forecast_steps,
                hidden_dim=self.hidden1,
                spei3_dropout=spei3_drop,
                base_dropout=dropout,
            )
            self.decoders = None
        else:
            self.cascade_decoder = None
            if self.quantiles:
                q_count = len(self.quantiles)
                if q_count < 3:
                    raise ValueError("quantiles requires at least three levels (e.g. [0.1, 0.5, 0.9])")
                if sorted(self.quantiles) != list(self.quantiles):
                    raise ValueError("quantiles must be strictly increasing")
                self.decoders = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(context_dim, self.hidden1), nn.ReLU(),
                        nn.Linear(self.hidden1, self.forecast_steps * q_count),
                    ) for _ in range(3)
                ])
                self.q_count = q_count
            else:
                self.decoders = nn.ModuleList([
                    ForecastDecoder(context_dim, self.forecast_steps, hidden_dim=self.hidden1, dropout=dropout)
                    for _ in range(3)
                ])

    @classmethod
    def from_config(cls, config: Union[dict, str, Path]) -> "DroughtCast":
        if isinstance(config, (str, Path)):
            import yaml
            cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        else:
            cfg = config
        return cls(cfg)

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.separate_encoders:
            return [encoder(x) for encoder in self.encoders]
        encoded = self.encoder(x)
        return [encoded, encoded, encoded]

    def attend(self, encoded_list: list[torch.Tensor], batch: int, steps: int, device) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor | None]:
        shared_context = shared_alpha = None
        if self.attentions is not None and self.attention_mode == "shared" and not self.separate_encoders:
            shared_context, shared_alpha = self.attentions[0](encoded_list[0])

        contexts, attentions = [], []
        for i, encoded in enumerate(encoded_list):
            if self.attentions is None:
                context = encoded[:, -1]
                alpha = torch.full((batch, steps, 1), 1.0 / steps, device=device)
            elif self.attention_mode == "shared":
                if self.separate_encoders:
                    context, alpha = self.attentions[0](encoded)
                else:
                    context, alpha = shared_context, shared_alpha
            else:
                context, alpha = self.attentions[i](encoded)
            contexts.append(context)
            attentions.append(alpha)
        return contexts, attentions, shared_alpha

    def forward(self, x: torch.Tensor, last_spei: torch.Tensor | None = None,
                static: torch.Tensor | None = None):
        """Forward pass.

        Args:
            x: [B, T, D] input window
            last_spei: optional [B, 3] last observed SPEI per scale (residual heads)
            static: optional [B, S] region-agnostic static descriptors
        """
        gate = None
        if self.variable_gate is not None:
            gate = self.variable_gate(x.mean(dim=1))
            x = x * gate.unsqueeze(1)

        encoded_list = self.encode(x)
        contexts, attentions, shared_alpha = self.attend(encoded_list, x.shape[0], x.shape[1], x.device)

        if self.static_context:
            if static is None:
                static = torch.zeros(x.shape[0], self._static_dim, device=x.device, dtype=x.dtype)
            contexts = [torch.cat([c, static], dim=-1) for c in contexts]

        preds: dict[str, torch.Tensor] = {}
        if self.cascade_decoder is not None and not self.quantiles:
            c_stack = torch.stack(contexts, dim=1)
            p3, p6, p12 = self.cascade_decoder(c_stack)
            preds = {"spei_3": p3, "spei_6": p6, "spei_12": p12}
        else:
            for i, (name, decoder) in enumerate(zip(self.SCALE_NAMES, self.decoders)):
                raw = decoder(contexts[i])
                if self.quantiles:
                    raw = raw.view(x.shape[0], self.forecast_steps, self.q_count)
                    # Monotone: median + softplus distances when 0.5 is present with 3 quantiles
                    if self.q_count == 3 and 0.5 in self.quantiles:
                        median = raw[..., 1]
                        lower = median - torch.nn.functional.softplus(raw[..., 0])
                        upper = median + torch.nn.functional.softplus(raw[..., 2])
                        preds[name] = torch.stack((lower, median, upper), dim=-1)
                    else:
                        preds[name] = raw
                else:
                    preds[name] = raw

        if self.head_mode == "residual":
            if last_spei is None:
                raise ValueError("head_mode=residual requires last_spei [B, 3]")
            for i, name in enumerate(self.SCALE_NAMES):
                base = last_spei[:, i].unsqueeze(-1)
                if self.quantiles:
                    preds[name] = preds[name] + base.unsqueeze(-1)
                else:
                    preds[name] = preds[name] + base

        attention_output = (
            shared_alpha if self.attention_mode == "shared" and not self.separate_encoders
            else torch.stack(attentions, dim=1)
        )
        if self.attentions is None:
            attention_output = torch.stack(attentions, dim=1)

        aux = {
            "attention": attention_output,
            "variable_gate": gate,
            "context": contexts[0] if contexts else None,
            "contexts": contexts,
        }
        return preds, aux

    def predict_tensor(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Stacked point predictions [B, 3, L] (median if quantiles)."""
        preds, _ = self.forward(x, **kwargs)
        if self.quantiles:
            mid = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
            return torch.stack([preds[k][..., mid] for k in self.SCALE_NAMES], dim=1)
        return torch.stack([preds[k] for k in self.SCALE_NAMES], dim=1)


# Deprecated aliases — prefer DroughtCast.
MultiScaleForecaster = DroughtCast
MultiHorizonMultiHeadForecaster = DroughtCast
