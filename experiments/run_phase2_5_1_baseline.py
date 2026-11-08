"""Run one instrumented Phase 2.5.1 train/validation baseline.

This deliberately calls the existing fit routine directly. It never constructs
test windows and never calls the Phase 2.2 runner, whose normal workflow includes
test evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
import yaml

from data.loader import load_site_table, validate_dataset_protocol
from data.protocol import DATASET_PROTOCOL_VERSION, MODEL_FEATURES, SPLIT_BOUNDARIES
from experiments.run_phase2_2_neural import (
    BASELINE_RUN,
    NORMALIZER_PATH,
    _fit_profile,
    _model_config,
    _verify_normalizer,
    seed_everything,
    sha256,
)
from experiments.training_diagnostics import TrainingDiagnostics

CONFIG_PATH = Path("configs/config.yaml")
NEURAL_RUN = Path(
    "results/maharashtra/phase2/neural/"
    "protocol10_seed42_e9dfa7ce76_ae290b4115_1d7a5ce0"
)
RESULT_ROOT = Path("results/maharashtra/phase2/diagnostics")
MODEL_ROOT = Path("models/diagnostics")
SEED = 42


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run() -> dict:
    config_bytes = CONFIG_PATH.read_bytes()
    base_config = yaml.safe_load(config_bytes)
    neural_metadata = _load_json(NEURAL_RUN / "neural_experiment_metadata.json")
    neural_hyperparameters = _load_json(NEURAL_RUN / "neural_hyperparameters.json")
    baseline_metadata = _load_json(BASELINE_RUN / "baseline_experiment_metadata.json")
    baseline_leakage = _load_json(BASELINE_RUN / "baseline_leakage_audit.json")

    dataset_path = Path(base_config["data"]["path"])
    dataset_hash = sha256(dataset_path)
    normalizer_hash = sha256(NORMALIZER_PATH)
    if dataset_hash != neural_metadata["dataset_sha256"]:
        raise RuntimeError("Dataset hash differs from the locked Phase 2.2 experiment")
    if normalizer_hash != neural_metadata["normalizer_sha256"]:
        raise RuntimeError("Normalizer hash differs from the locked Phase 2.2 experiment")
    if dataset_hash != baseline_metadata["dataset_sha256"]:
        raise RuntimeError("Dataset hash differs from the Phase 2.1 baseline")
    if normalizer_hash != baseline_metadata["normalizer_sha256"]:
        raise RuntimeError("Normalizer hash differs from the Phase 2.1 baseline")
    if base_config["data"]["features"] != MODEL_FEATURES:
        raise RuntimeError("Configured feature order differs from protocol 10")
    if base_config["data"]["split_boundaries"] != SPLIT_BOUNDARIES:
        raise RuntimeError("Configured split boundaries differ from protocol 10")
    if neural_metadata["feature_order"] != MODEL_FEATURES:
        raise RuntimeError("Phase 2.2 feature order differs from protocol 10")
    if neural_metadata["split_boundaries"] != SPLIT_BOUNDARIES:
        raise RuntimeError("Phase 2.2 split boundaries differ from protocol 10")

    dataset_metadata_path = dataset_path.parent / "dataset_metadata.json"
    validate_dataset_protocol(
        dataset_metadata_path,
        base_config["data"].get("target_source", "local"),
        list(MODEL_FEATURES),
    )
    normalizer = _load_json(NORMALIZER_PATH)
    _verify_normalizer(normalizer, normalizer_hash, dataset_hash, base_config)

    script_path = Path(__file__)
    run_fingerprint = hashlib.sha256(
        f"{sha256(script_path)}:{dataset_hash}:{normalizer_hash}:{SEED}".encode()
    ).hexdigest()[:12]
    run_id = f"phase2_5_1_protocol{DATASET_PROTOCOL_VERSION}_seed{SEED}_{run_fingerprint}"
    result_dir = RESULT_ROOT / run_id
    checkpoint_dir = MODEL_ROOT / run_id
    if result_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic run {run_id}")
    result_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)

    # The loader truncates the source frame at validation end before quality
    # checks and window construction. Test windows are never materialized.
    datasets = load_site_table(
        dataset_path,
        features=list(MODEL_FEATURES),
        seq_len=6,
        forecast_steps=4,
        metadata_path=NORMALIZER_PATH,
        site_filter=list(base_config["data"]["site_filter"]),
        site_specs=base_config.get("sites"),
        split_config=SPLIT_BOUNDARIES,
        include_splits=("train", "val"),
    )
    if set(datasets) != {"train", "val", "test"} or len(datasets["test"]) != 0:
        raise RuntimeError(
            "Loader must return only an empty test placeholder when asked for train/val; "
            f"keys={sorted(datasets)}, test_windows={len(datasets.get('test', []))}"
        )
    train_data, val_data = datasets["train"], datasets["val"]
    if len(train_data) != 2916 or len(val_data) != 468:
        raise RuntimeError(
            f"Window counts differ from locked protocol: train={len(train_data)}, "
            f"validation={len(val_data)}"
        )
    if not torch.isfinite(train_data.X).all() or not torch.isfinite(val_data.X).all():
        raise RuntimeError("Non-finite model inputs in train/validation windows")
    if any(max(row["target_dates"]) > SPLIT_BOUNDARIES["val_end"] for row in val_data.metadata):
        raise RuntimeError("Validation target extends past the configured validation boundary")

    selected_profile = neural_hyperparameters["selected_profile"]
    profile = neural_hyperparameters["selected_parameters"]
    if selected_profile != "regularized_batch64":
        raise RuntimeError(f"Unexpected frozen profile: {selected_profile}")
    effective_config = _model_config(base_config, profile)
    diagnostics = TrainingDiagnostics(
        result_dir,
        run_id=run_id,
        profile=selected_profile,
        seed=SEED,
        effective_config=effective_config,
        extreme_weight=float(effective_config["training"].get("extreme_weight", 1.0)),
        extreme_threshold=float(effective_config["training"].get("extreme_threshold", -1.0)),
    )

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = checkpoint_dir / "selected_seed_42.pt"
    model, selection, history = _fit_profile(
        base_config,
        selected_profile,
        profile,
        train_data,
        val_data,
        sorted(base_config["data"]["site_filter"]),
        device,
        SEED,
        checkpoint_path,
        diagnostics,
    )
    del model

    pd.DataFrame(history).to_csv(result_dir / "training_history.csv", index=False)
    metadata = {
        "experiment": "Phase 2.5.1 instrumented existing-objective baseline",
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "profile": selected_profile,
        "effective_config": effective_config,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "normalizer_path": str(NORMALIZER_PATH),
        "normalizer_sha256": normalizer_hash,
        "protocol_version": DATASET_PROTOCOL_VERSION,
        "features": list(MODEL_FEATURES),
        "split_boundaries": SPLIT_BOUNDARIES,
        "materialized_splits": ["train", "val"],
        "test_placeholder_windows": len(datasets["test"]),
        "window_counts": {"train": len(train_data), "val": len(val_data)},
        "test_split_materialized_or_evaluated": False,
        "training_objective": "existing sum of three scale-wise extreme-weighted quantile pinball losses",
        "checkpoint_selection": "existing validation equal-site SPEI-3 T+3/T+4 MSE skill vs persistence",
        "selection": selection,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "source_hashes": {
            "diagnostic_runner": sha256(script_path),
            "phase2_2_runner": sha256(Path("experiments/run_phase2_2_neural.py")),
            "observer": sha256(Path("experiments/training_diagnostics.py")),
            "config": sha256(CONFIG_PATH),
            "dataset_metadata": sha256(dataset_metadata_path),
            "normalizer": normalizer_hash,
        },
        "baseline_protocol_hash_match": True,
        "phase2_1_leakage_audit_used_for_preflight_only": True,
        "baseline_flags": {
            "test_used_for_fit_or_selection": baseline_metadata.get("test_used_for_fit_or_selection"),
            "baseline_test_already_consumed": baseline_metadata.get("test_evaluation_consumed"),
            "required_leakage_flags_pass": all(
                baseline_leakage.get(key, False)
                for key in (
                    "no_cross_split_window_detected",
                    "no_future_targets_in_input_windows",
                    "no_future_climate_index_dates",
                )
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
    }
    _write_json(result_dir / "experiment_metadata.json", metadata)
    _write_json(result_dir / "effective_configuration.json", effective_config)
    _write_json(result_dir / "selection.json", selection)
    if sha256(dataset_path) != dataset_hash or sha256(NORMALIZER_PATH) != normalizer_hash:
        raise RuntimeError("Canonical data or normalizer hash changed during diagnostic run")
    if CONFIG_PATH.read_bytes() != config_bytes:
        raise RuntimeError("Configuration changed during diagnostic run")
    return {"run_id": run_id, "result_dir": str(result_dir), "checkpoint_dir": str(checkpoint_dir)}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
