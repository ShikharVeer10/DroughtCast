"""Reproducibility metadata and checkpoint compatibility helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from data.protocol import DATASET_PROTOCOL_VERSION


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(config: dict) -> str:
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def architecture_fingerprint(config: dict) -> str:
    model = config.get("model", {})
    data = config.get("data", {})
    fields = {
        "input_dim": data.get("input_dim"),
        "features": list(data.get("features") or []),
        "seq_len": data.get("seq_len"),
        "forecast_steps": data.get("forecast_steps"),
        "target_scales": list(data.get("target_scales") or ["spei_3", "spei_6", "spei_12"]),
        "target_source": data.get("target_source", "local"),
        "period": [data.get("start_year"), data.get("end_year"), data.get("end_month", 12)],
        "encoder": model.get("encoder", "lstm"),
        "hidden_dim_1": model.get("hidden_dim_1", 64),
        "hidden_dim_2": model.get("hidden_dim_2", 32),
        "attention": model.get("attention", "shared"),
        "heads": model.get("heads", model.get("head_mode", "direct")),
        "quantiles": model.get("quantiles"),
        "static_context": model.get("static_context", False),
        "use_cascaded_decoder": model.get("use_cascaded_decoder", False),
        "use_attention": model.get("use_attention", True),
        "use_variable_gate": model.get("use_variable_gate", False),
    }
    return config_hash(fields)


def artifact_metadata(
    config: dict, dataset_path: str | Path | None = None, seed: int | None = None
) -> dict:
    return {
        "git_commit": git_commit(),
        "config_hash": config_hash(config),
        "architecture_fingerprint": architecture_fingerprint(config),
        "dataset_path": str(dataset_path) if dataset_path else None,
        "dataset_schema_version": DATASET_PROTOCOL_VERSION,
        "dataset_sha256": sha256_file(dataset_path)
        if dataset_path and Path(dataset_path).is_file()
        else None,
        "seed": seed,
        "python": sys.version,
        "torch": torch.__version__,
        "features": list(config.get("data", {}).get("features") or []),
        "forecast_steps": config.get("data", {}).get("forecast_steps"),
        "seq_len": config.get("data", {}).get("seq_len"),
        "sites": sorted((config.get("sites") or {}).keys()),
        "split_boundaries": dict(config.get("data", {}).get("split_boundaries") or {}),
        "target_scales": list(config.get("data", {}).get("target_scales") or ["spei_3", "spei_6", "spei_12"]),
        "target_source": config.get("data", {}).get("target_source", "local"),
        "period": [
            config.get("data", {}).get("start_year"),
            config.get("data", {}).get("end_year"),
            config.get("data", {}).get("end_month", 12),
        ],
        "quantiles": list(config.get("model", {}).get("quantiles") or []),
    }


def checkpoint_metadata_path(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path).with_suffix(".metadata.json")


def load_checkpoint(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Load tensor-only checkpoint weights and merge the JSON provenance sidecar."""
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint must contain a mapping: {checkpoint_path}")
    metadata_path = checkpoint_metadata_path(checkpoint_path)
    if metadata_path.is_file():
        sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
        state.update({key: value for key, value in sidecar.items() if key != "checkpoint_sha256"})
        expected_hash = sidecar.get("checkpoint_sha256")
        if expected_hash and sha256_file(checkpoint_path) != expected_hash:
            raise ValueError(f"Checkpoint was modified after its metadata was written: {checkpoint_path}")
    return state


def validate_checkpoint_metadata(
    checkpoint: dict, config: dict, dataset_path: str | Path | None = None
) -> None:
    if "artifact_metadata" not in checkpoint:
        raise ValueError(
            "Checkpoint has no reproducibility metadata and is considered legacy. "
            "Retrain it with the current train.py before evaluation or inference."
        )
    checkpoint_schema = checkpoint["artifact_metadata"].get("dataset_schema_version")
    if checkpoint_schema != DATASET_PROTOCOL_VERSION:
        raise ValueError(
            "Checkpoint dataset schema is incompatible with the current dataset protocol "
            f"(checkpoint={checkpoint_schema}, current={DATASET_PROTOCOL_VERSION}). "
            "Rebuild the dataset and retrain the checkpoint."
        )
    expected = architecture_fingerprint(config)
    actual = checkpoint.get("artifact_metadata", {}).get("architecture_fingerprint")
    if actual and actual != expected:
        raise ValueError(
            "Checkpoint architecture is incompatible with the supplied config "
            f"(checkpoint={actual}, config={expected})."
        )
    artifact = checkpoint["artifact_metadata"]
    checks = {
        "features": list(config.get("data", {}).get("features") or []),
        "seq_len": config.get("data", {}).get("seq_len"),
        "forecast_steps": config.get("data", {}).get("forecast_steps"),
        "sites": sorted((config.get("sites") or {}).keys()),
        "target_scales": list(config.get("data", {}).get("target_scales") or ["spei_3", "spei_6", "spei_12"]),
        "quantiles": list(config.get("model", {}).get("quantiles") or []),
        "split_boundaries": dict(config.get("data", {}).get("split_boundaries") or {}),
        "target_source": config.get("data", {}).get("target_source", "local"),
        "period": [
            config.get("data", {}).get("start_year"),
            config.get("data", {}).get("end_year"),
            config.get("data", {}).get("end_month", 12),
        ],
    }
    for key, expected_value in checks.items():
        if artifact.get(key) != expected_value:
            raise ValueError(
                f"Checkpoint {key} do not match the supplied config "
                f"(checkpoint={artifact.get(key)!r}, config={expected_value!r})."
            )
    if dataset_path is not None:
        expected_hash = sha256_file(dataset_path)
        if artifact.get("dataset_sha256") != expected_hash:
            raise ValueError(
                "Checkpoint dataset hash does not match the dataset supplied for evaluation/inference."
            )
