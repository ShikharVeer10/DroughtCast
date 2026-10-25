"""Allowed configuration keys and validation helpers."""
from __future__ import annotations

from typing import Any

ALLOWED_TOP = {"experiment", "sites", "data", "model", "training"}

ALLOWED_EXPERIMENT = {"name", "seed"}
ALLOWED_DATA = {
    "path", "metadata_path", "start_year", "end_year", "features", "seq_len",
    "forecast_steps", "input_dim", "site_filter",
}
ALLOWED_MODEL = {
    "input_dim", "hidden_dim_1", "hidden_dim_2", "dropout", "attention",
    "encoder", "heads", "head_mode", "use_cascaded_decoder", "variable_gate",
    "separate_encoders", "quantiles", "spei3_dropout", "static_context",
    "input_extras", "consistency_weight",
}
ALLOWED_TRAINING = {
    "batch_size", "epochs", "learning_rate", "weight_decay",
    "early_stopping_patience", "lr_patience", "extreme_weight",
    "extreme_threshold", "checkpoint_dir", "loss", "task_weighting",
}

# Keys removed from the diagram config; reject if present.
DEAD_MODEL_KEYS = {
    "hidden_dim", "out_hidden_dim", "num_layers", "num_quantiles",
    "use_teleconnection_gate", "use_multi_horizon_attn", "use_cascade_decoder",
    "forecast_steps",  # lives under data
}


def _reject_unknown(section: str, values: dict[str, Any] | None, allowed: set[str]) -> None:
    if not values:
        return
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            f"Unknown config key(s) under '{section}': {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
        )


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Reject unknown / dead keys with a clear error. Returns cfg unchanged."""
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
