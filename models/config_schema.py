"""Configuration validation for the DroughtCast forecaster."""

from __future__ import annotations

from typing import Any

ALLOWED_TOP = {"experiment", "sites", "data", "model", "training"}
ALLOWED_EXPERIMENT = {"name", "seed"}
ALLOWED_DATA = {
    "path", "metadata_path", "start_year", "end_year", "end_month", "features", "seq_len",
    "forecast_steps", "input_dim", "site_filter", "train_split", "val_split", "test_split", "target_scales", "split_boundaries",
}
ALLOWED_MODEL = {
    "input_dim", "hidden_dim_1", "hidden_dim_2", "dropout", "attention", "encoder",
    "heads", "head_mode", "use_cascaded_decoder", "variable_gate", "separate_encoders",
    "quantiles", "spei3_dropout", "static_context", "input_extras", "consistency_weight",
    "use_attention", "use_variable_gate",
}
ALLOWED_TRAINING = {
    "batch_size", "epochs", "learning_rate", "lr", "weight_decay", "early_stopping_patience",
    "patience", "lr_patience", "extreme_weight", "extreme_threshold", "grad_clip",
    "checkpoint_dir", "loss", "task_weighting",
}
DEAD_MODEL_KEYS = {"hidden_dim", "out_hidden_dim", "num_layers", "num_quantiles", "forecast_steps"}


def _reject_unknown(section: str, values: dict[str, Any] | None, allowed: set[str]) -> None:
    if not values:
        return
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown config key(s) under '{section}': {sorted(unknown)}")


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise TypeError("config must be a mapping")
    unknown = set(cfg) - ALLOWED_TOP
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    _reject_unknown("experiment", cfg.get("experiment"), ALLOWED_EXPERIMENT)
    _reject_unknown("data", cfg.get("data"), ALLOWED_DATA)
    _reject_unknown("model", cfg.get("model"), ALLOWED_MODEL)
    _reject_unknown("training", cfg.get("training"), ALLOWED_TRAINING)
    data, model = cfg.get("data", {}), cfg.get("model", {})
    features = data.get("features")
    if features is not None:
        if not isinstance(features, list) or not features:
            raise ValueError("data.features must be a non-empty list")
        if model.get("input_dim") is not None and model["input_dim"] != len(features):
            raise ValueError(f"model.input_dim ({model['input_dim']}) must equal len(data.features) ({len(features)})")
    quantiles = model.get("quantiles")
    if quantiles is not None and (len(quantiles) != 3 or quantiles != sorted(quantiles) or any(not 0 < q < 1 for q in quantiles)):
        raise ValueError("model.quantiles must be three increasing probabilities in (0,1)")
    return cfg
