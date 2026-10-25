"""Joint multi-scale SPEI forecaster, with switchable encoder and attention."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from models.attention import TemporalAttention, MultiHorizonTemporalAttention
from models.encoder import SharedLSTMEncoder, TeleconnectionGatedEncoder
from models.decoder import ForecastDecoder, MultiHorizonMultiHeadDecoder, CascadedHydrologicalDecoder


class MultiScaleForecaster(nn.Module):
    def __init__(self, config):
        super().__init__()
        data, model = config["data"], config["model"]
        input_dim = data.get("input_dim")
        if not input_dim and "features" in data:
            input_dim = len(data["features"])
            data["input_dim"] = input_dim
        if not input_dim:
            raise ValueError("data.input_dim must be inferred from dataset feature columns or provided in config['data']['input_dim']")
        hidden1, hidden2 = model.get("hidden_dim_1", 64), model.get("hidden_dim_2", 32)
        encoder_type = model.get("encoder", "lstm").lower()
        if encoder_type not in ("lstm", "gru", "transformer"):
            raise ValueError(f"Unknown encoder: {encoder_type}")
        if encoder_type == "transformer":
            self.input_projection = nn.Linear(input_dim, hidden2)

        def make_encoder():
            if encoder_type == "lstm":
                return SharedLSTMEncoder(input_dim if encoder_type != "transformer" else hidden2,
                                         hidden1, hidden2, model.get("dropout", .2))
            if encoder_type == "gru":
                return nn.GRU(input_dim, hidden2, batch_first=True)
            layer = nn.TransformerEncoderLayer(hidden2, nhead=4, batch_first=True,
                                               dropout=model.get("dropout", .2))
            return nn.TransformerEncoder(layer, num_layers=2)

        self.separate_encoders = bool(model.get("separate_encoders", False))
        if self.separate_encoders:
            self.encoders = nn.ModuleList([make_encoder() for _ in range(3)])
        else:
            self.encoder = make_encoder()
        attention_mode = model.get("attention", "shared")
        self.attention_mode = attention_mode
        if attention_mode == "none":
            self.attentions = None
        elif attention_mode == "shared":
            self.attentions = nn.ModuleList([TemporalAttention(hidden2)])
        elif attention_mode == "head_specific":
            self.attentions = nn.ModuleList([TemporalAttention(hidden2) for _ in range(3)])
        else:
            raise ValueError(f"Unknown attention mode: {attention_mode}")
        self.variable_gate = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Softmax(dim=-1)) if model.get("variable_gate", False) else None
        
        # Check whether cascaded hydrological conditioning is configured
        self.use_cascading = model.get("use_cascaded_decoder", False)
        spei3_drop = model.get("spei3_dropout", 0.35 if self.use_cascading else model.get("dropout", 0.2))
        
        if self.use_cascading:
            self.cascade_decoder = CascadedHydrologicalDecoder(
                context_dim=hidden2,
                forecast_steps=data["forecast_steps"],
                hidden_dim=hidden1,
                spei3_dropout=spei3_drop,
                base_dropout=model.get("dropout", 0.2)
            )
            self.decoders = None
        else:
            self.cascade_decoder = None
            self.decoders = nn.ModuleList([ForecastDecoder(hidden2, data["forecast_steps"], hidden_dim=hidden1, dropout=model.get("dropout", 0.2)) for _ in range(3)])

        self.quantiles = model.get("quantiles", None)
        if self.quantiles:
            q_count = len(self.quantiles)
            # Monotone parameterization: median plus nonnegative distances to
            # the lower/upper quantiles. This prevents crossing by construction.
            if q_count != 3:
                raise ValueError("Quantile mode currently requires exactly three quantiles (lower, median, upper)")
            if not (0 < self.quantiles[0] < self.quantiles[1] < self.quantiles[2] < 1) or self.quantiles[1] != 0.5:
                raise ValueError("quantiles must be strictly increasing and have 0.5 as the median")
            self.decoders = nn.ModuleList([nn.Sequential(nn.Linear(hidden2, 32), nn.ReLU(),
                nn.Linear(32, data["forecast_steps"] * 3)) for _ in range(3)])
            self.q_count = q_count

    @classmethod
    def from_config(cls, config: Union[dict, str, Path]) -> MultiScaleForecaster:
        """Instantiate MultiScaleForecaster directly from a dictionary or YAML configuration file path."""
        if isinstance(config, (str, Path)):
            import yaml
            cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        else:
            cfg = config
        return cls(cfg)

    def forward(self, x):
        gate = None
        if self.variable_gate is not None:
            gate = self.variable_gate(x.mean(dim=1))
            x = x * gate.unsqueeze(1)
        if hasattr(self, "input_projection"):
            x = self.input_projection(x)
        if self.separate_encoders:
            encoded_list = [encoder(x) for encoder in self.encoders]
        else:
            encoded_list = [self.encoder(x)] * 3
        encoded_list = [encoded[0] if isinstance(encoded, tuple) else encoded for encoded in encoded_list]
        preds, attentions = {}, []
        names = ("spei_3", "spei_6", "spei_12")
        shared_context = shared_alpha = None
        if self.attentions is not None and self.attention_mode == "shared" and not self.separate_encoders:
            shared_context, shared_alpha = self.attentions[0](encoded_list[0])

        contexts = []
        for i, name in enumerate(names):
            encoded = encoded_list[i]
            if self.attentions is None:
                context = encoded[:, -1]
                alpha = torch.full((x.shape[0], x.shape[1], 1), 1 / x.shape[1], device=x.device)
            elif self.attention_mode == "shared":
                if self.separate_encoders:
                    context, alpha = self.attentions[0](encoded)
                else:
                    context, alpha = shared_context, shared_alpha
            else:
                attention = self.attentions[i]
                context, alpha = attention(encoded)
            contexts.append(context)
            attentions.append(alpha)

        if self.cascade_decoder is not None and not self.quantiles:
            c_stack = torch.stack(contexts, dim=1) # [B, 3, H]
            p3, p6, p12 = self.cascade_decoder(c_stack)
            preds["spei_3"] = p3
            preds["spei_6"] = p6
            preds["spei_12"] = p12
        else:
            for i, (name, decoder) in enumerate(zip(names, self.decoders)):
                context = contexts[i]
                raw = decoder(context)
                if self.quantiles:
                    raw = raw.view(x.shape[0], 3, -1).transpose(1, 2)
                    median = raw[..., 1]
                    lower = median - torch.nn.functional.softplus(raw[..., 0])
                    upper = median + torch.nn.functional.softplus(raw[..., 2])
                    preds[name] = torch.stack((lower, median, upper), dim=-1)
                else:
                    preds[name] = raw

        attention_output = (shared_alpha if self.attention_mode == "shared" and not self.separate_encoders
                            else torch.stack(attentions, dim=1))
        if self.attentions is None:
            attention_output = torch.stack(attentions, dim=1)
        return preds, {"attention": attention_output, "variable_gate": gate}


# Kept for compatibility with earlier checkpoints/scripts; new code should use MultiScaleForecaster.
MultiHorizonMultiHeadForecaster = MultiScaleForecaster
