"""Reproducibility metadata and checkpoint compatibility helpers."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import torch
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
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
        "forecast_steps": data.get("forecast_steps"),
        "encoder": model.get("encoder", "lstm"),
        "hidden_dim_1": model.get("hidden_dim_1", 64),
        "hidden_dim_2": model.get("hidden_dim_2", 32),
        "attention": model.get("attention", "shared"),
        "heads": model.get("heads", model.get("head_mode", "direct")),
        "quantiles": model.get("quantiles"),
        "static_context": model.get("static_context", False),
        "use_cascaded_decoder": model.get("use_cascaded_decoder", False),
    }
    return config_hash(fields)


def artifact_metadata(config: dict, dataset_path: str | Path | None = None,
                      seed: int | None = None) -> dict:
    return {
        "git_commit": git_commit(),
        "config_hash": config_hash(config),
        "architecture_fingerprint": architecture_fingerprint(config),
        "dataset_path": str(dataset_path) if dataset_path else None,
        "dataset_sha256": sha256_file(dataset_path) if dataset_path and Path(dataset_path).is_file() else None,
        "seed": seed,
        "python": sys.version,
        "torch": torch.__version__,
        "features": list(config.get("data", {}).get("features") or []),
        "forecast_steps": config.get("data", {}).get("forecast_steps"),
        "seq_len": config.get("data", {}).get("seq_len"),
    }


def validate_checkpoint_metadata(checkpoint: dict, config: dict) -> None:
    if "artifact_metadata" not in checkpoint:
        raise ValueError(
            "Checkpoint has no reproducibility metadata and is considered legacy. "
            "Retrain it with the current train.py before evaluation or inference."
        )
    expected = architecture_fingerprint(config)
    actual = checkpoint.get("artifact_metadata", {}).get("architecture_fingerprint")
    if actual and actual != expected:
        raise ValueError(
            "Checkpoint architecture is incompatible with the supplied config "
            f"(checkpoint={actual}, config={expected})."
        )
