from __future__ import annotations

from typing import Any

ALLOWED_TOP = {"experiment", "sites", "data", "model", "training"}

ALLOWED_EXPERIMENT = {"name", "seed"}
ALLOWED_DATA = {
    "path",
    "metadata_path",
    "start_year",
    "end_year",
    "features",
    "seq_len",
    "forecast_steps",
    "input_dim",
    "site_filter",
}
ALLOWED_MODEL = {
    "input_dim",
    "hidden_dim_1",
    "hidden_dim_2",
    "dropout",
    "attention",
    "encoder",
    "heads",
    "head_mode",
    "use_cascaded_decoder",
    "variable_gate",
    "separate_encoders",
    "quantiles",
    "spei3_dropout",
    "static_context",
    "input_extras",
    "consistency_weight",
}
ALLOWED_TRAINING = {
    "batch_size",
    "epochs",
    "learning_rate",
    "weight_decay",
    "early_stopping_patience",
    "lr_patience",
    "extreme_weight",
    "extreme_threshold",
    "checkpoint_dir",
    "loss",
    "task_weighting",
}

canonical_features = [
        "PRECTOTCORR",
        "T2M",
        "T2MDEW",
        "WS10M",
        "ALLSKY_SFC_SW_DWN",
        "GWETTOP",
        "GWETROOT",
        "VPD",
        "T2M_RANGE",
        "NDVI",
        "VCI",
        "oni",
        "dmi",
        "nao",
        "pdo",
        "spei_3_obs",
        "spei_6_obs",
        "spei_12_obs",
    ]
    features = data.get("features")
    if not isinstance(features, list) or not features:
      raise ValueError("data.features must be a non-empty list")
    unknown_feats = [f for f in features if f not in canonical_features]
    if unknown_feats:
      raise ValueError(f"Unknown features in data.features: {unknown_feats}")
    if model.get("input_dim") != len(features):
      raise ValueError(
          f"model.input_dim ({model.get('input_dim')}) must equal"
          f" len(data.features) ({len(features)})"
      )

DEAD_MODEL_KEYS = {
    "hidden_dim",
    "out_hidden_dim",
    "num_layers",
    "num_quantiles",
    "use_teleconnection_gate",
    "use_multi_horizon_attn",
    "use_cascade_decoder",
    "forecast_steps",  # lives under data
}


def _reject_unknown(section: str, values: dict[str, Any] | None, allowed: set[str]) -> None:
    if not values:
        return
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown config key(s) under '{section}': {sorted(unknown)}. "f"Allowed: {sorted(allowed)}")


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise TypeError("config must be a mapping")
    unknown_top = set(cfg) - ALLOWED_TOP
    if unknown_top:
        raise ValueError(f"Unknown top-level config key(s): {sorted(unknown_top)}")
    _reject_unknown("experiment", cfg.get("experiment"), ALLOWED_EXPERIMENT)
    _reject_unknown("data", cfg.get("data"), ALLOWED_DATA)
    _reject_unknown("training", cfg.get("training"), ALLOWED_TRAINING)
    model = cfg.get("model") or {}
    dead = set(model) & DEAD_MODEL_KEYS
    if dead:
        raise ValueError(
            f"Dead/removed model config key(s): {sorted(dead)}. "
            "Use hidden_dim_1/hidden_dim_2, attention, heads/head_mode, "
            "use_cascaded_decoder instead of the legacy flags."
        )
    _reject_unknown("model", model, ALLOWED_MODEL)
    return cfg
