"""Run the train/validation-only Phase 2.5.1B objective comparison."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from data.loader import load_site_table, validate_dataset_protocol
from data.protocol import DATASET_PROTOCOL_VERSION, MODEL_FEATURES, SPLIT_BOUNDARIES
from experiments.run_phase2_2_neural import (
    NORMALIZER_PATH,
    QUANTILES,
    TARGETS,
    ValidationEarlyStopper,
    _evaluate_predictions,
    _model_config,
    _primary_site_mean_skill,
    _quantile_predictions,
    _verify_normalizer,
    seed_everything,
    sha256,
)
from experiments.training_diagnostics import TrainingDiagnostics
from models.forecaster import MultiScaleForecaster
from train import pinball_loss

CONFIG_PATH = Path("configs/config.yaml")
PHASE22_RUN = Path(
    "results/maharashtra/phase2/neural/"
    "protocol10_seed42_e9dfa7ce76_ae290b4115_1d7a5ce0"
)
PHASE251_BASELINE = Path(
    "results/maharashtra/phase2/diagnostics/"
    "phase2_5_1_protocol10_seed42_f5cd5beb9f59"
)
PROFILE_NAME = "regularized_batch64"
SEED = 42
PRIMARY_MSE_WEIGHT = 0.5
EARLY_STOPPING_PATIENCE = 12
LR_PATIENCE = 5


def primary_mse_loss(prediction, target, mask):
    """Masked q50 MSE for SPEI-3 T+3/T+4; target/prediction shape [batch, 2]."""
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Primary prediction, target, and mask shapes must match")
    valid = mask.to(device=target.device, dtype=torch.bool)
    if torch.any(valid & (~torch.isfinite(target) | ~torch.isfinite(prediction))):
        raise ValueError("Primary MSE requires finite predictions and targets at valid cells")
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    squared_error = (safe_prediction - safe_target).square()
    return (squared_error * valid.to(squared_error.dtype)).sum() / valid.sum().clamp_min(1)


def combined_objective(predictions, targets, mask, config):
    """Return existing scale-summed pinball plus the predeclared primary MSE term."""
    scale_losses = [
        pinball_loss(
            predictions[scale],
            targets[:, index],
            mask[:, index],
            QUANTILES,
            float(config["training"].get("extreme_weight", 1.0)),
            float(config["training"].get("extreme_threshold", -1.0)),
        )
        for index, scale in enumerate(TARGETS)
    ]
    pinball_total = torch.stack(scale_losses).sum()
    median = predictions["spei_3"][:, 2:4, 1]
    mse_primary = primary_mse_loss(median, targets[:, 0, 2:4], mask[:, 0, 2:4])
    total = pinball_total + PRIMARY_MSE_WEIGHT * mse_primary
    return total, pinball_total, mse_primary


def _train_epoch(model, loader, optimizer, device, config, observer, epoch):
    model.train()
    totals = {"total": 0.0, "pinball": 0.0, "primary_mse": 0.0}
    count = 0
    for x, y, mask in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device)
        optimizer.zero_grad(set_to_none=True)
        predictions, _ = model(x)
        loss, pinball, primary_mse = combined_objective(predictions, y, mask, config)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite alternative objective")
        observer.add_predictions("train", epoch, predictions, y, mask)
        loss.backward()
        snapshot = observer.before_optimizer_step(model)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"].get("grad_clip", 1.0))
        )
        optimizer.step()
        observer.after_optimizer_step(model, snapshot)
        batch_size = len(x)
        totals["total"] += float(loss.detach().cpu()) * batch_size
        totals["pinball"] += float(pinball.detach().cpu()) * batch_size
        totals["primary_mse"] += float(primary_mse.detach().cpu()) * batch_size
        count += batch_size
    denominator = max(count, 1)
    return {name: value / denominator for name, value in totals.items()}


def _pooled_primary_mse(dataset, predictions):
    y = dataset.Y.numpy()[:, 0, 2:4].astype(float)
    mask = dataset.mask.numpy()[:, 0, 2:4].astype(bool)
    median = predictions["spei_3"][:, 2:4, 1].astype(float)
    valid = mask & np.isfinite(y) & np.isfinite(median)
    if not valid.any():
        return None, 0
    return float(np.mean(np.square(y[valid] - median[valid]))), int(valid.sum())


def _assert_preflight(base_config, dataset_path, dataset_hash, normalizer_hash):
    phase22 = json.loads((PHASE22_RUN / "neural_experiment_metadata.json").read_text())
    hyper = json.loads((PHASE22_RUN / "neural_hyperparameters.json").read_text())
    baseline = json.loads((PHASE251_BASELINE / "experiment_metadata.json").read_text())
    baseline_integrity = json.loads((PHASE251_BASELINE / "source_and_data_hashes.json").read_text())
    if dataset_hash != phase22["dataset_sha256"] or dataset_hash != baseline["dataset_sha256"]:
        raise RuntimeError("Dataset hash differs from the locked Phase 2.2/2.5.1 runs")
    if normalizer_hash != phase22["normalizer_sha256"] or normalizer_hash != baseline["normalizer_sha256"]:
        raise RuntimeError("Normalizer hash differs from the locked Phase 2.2/2.5.1 runs")
    if not baseline_integrity.get("effective_configuration_matches_phase2_2_after_legacy_gate_alias_removal"):
        raise RuntimeError("Phase 2.5.1 baseline effective config did not match Phase 2.2")
    if base_config["data"]["features"] != MODEL_FEATURES:
        raise RuntimeError("Feature order differs from protocol 10")
    if base_config["data"]["split_boundaries"] != SPLIT_BOUNDARIES:
        raise RuntimeError("Split boundaries differ from protocol 10")
    if phase22["feature_order"] != MODEL_FEATURES or phase22["split_boundaries"] != SPLIT_BOUNDARIES:
        raise RuntimeError("Locked Phase 2.2 metadata does not match protocol 10")
    selected = hyper["selected_parameters"]
    if hyper["selected_profile"] != PROFILE_NAME:
        raise RuntimeError("Unexpected frozen Phase 2.2 profile")
    dataset_metadata = dataset_path.parent / "dataset_metadata.json"
    validate_dataset_protocol(
        dataset_metadata,
        base_config["data"].get("target_source", "local"),
        list(MODEL_FEATURES),
    )
    normalizer = json.loads(NORMALIZER_PATH.read_text(encoding="utf-8"))
    _verify_normalizer(normalizer, normalizer_hash, dataset_hash, base_config)
    return selected, phase22, baseline, dataset_metadata


def run() -> dict:
    config_bytes = CONFIG_PATH.read_bytes()
    base_config = yaml.safe_load(config_bytes)
    dataset_path = Path(base_config["data"]["path"])
    dataset_hash = sha256(dataset_path)
    normalizer_hash = sha256(NORMALIZER_PATH)
    profile, phase22, baseline_meta, dataset_metadata = _assert_preflight(
        base_config, dataset_path, dataset_hash, normalizer_hash
    )
    effective_config = _model_config(base_config, profile)
    effective_config["training"]["epochs"] = int(base_config["training"]["epochs"])
    effective_config["training"]["early_stopping_patience"] = EARLY_STOPPING_PATIENCE
    effective_config["training"]["lr_patience"] = LR_PATIENCE

    source_path = Path(__file__)
    run_key = hashlib.sha256(
        f"{sha256(source_path)}:{dataset_hash}:{normalizer_hash}:{SEED}:mse{PRIMARY_MSE_WEIGHT}".encode()
    ).hexdigest()[:12]
    run_id = f"phase2_5_1b_protocol10_seed{SEED}_{run_key}"
    result_dir = Path("results/maharashtra/phase2/diagnostics") / run_id
    model_dir = Path("models/diagnostics") / run_id
    if result_dir.exists() or model_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 2.5.1B run {run_id}")
    result_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    # This loader truncates at validation end before creating windows.
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
    if len(datasets["train"]) != 2916 or len(datasets["val"]) != 468 or len(datasets["test"]) != 0:
        raise RuntimeError("Train/validation window counts differ or test placeholder is non-empty")
    train_dataset, validation_dataset = datasets["train"], datasets["val"]
    sites = sorted(base_config["data"]["site_filter"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    seed_everything(SEED)
    model = MultiScaleForecaster(effective_config).to(device)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(train_dataset.X, train_dataset.Y, train_dataset.mask),
        batch_size=int(profile["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(profile["learning_rate"]),
        weight_decay=float(profile["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=LR_PATIENCE, factor=0.5
    )
    stopper = ValidationEarlyStopper(EARLY_STOPPING_PATIENCE)
    checkpoint_path = model_dir / "selected_seed_42.pt"
    observer = TrainingDiagnostics(
        result_dir,
        run_id=run_id,
        profile=f"{PROFILE_NAME}_plus_primary_mse_0.5",
        seed=SEED,
        effective_config={
            **effective_config,
            "diagnostic_objective": {
                "definition": "L_pinball + 0.5 * pooled valid SPEI-3 q50 MSE at T+3/T+4",
                "primary_mse_weight": PRIMARY_MSE_WEIGHT,
            },
        },
        extreme_weight=float(effective_config["training"].get("extreme_weight", 1.0)),
        extreme_threshold=float(effective_config["training"].get("extreme_threshold", -1.0)),
    )
    history = []
    best_state = None
    selected_skill_by_site = None
    max_epochs = int(effective_config["training"]["epochs"])
    for epoch in range(1, max_epochs + 1):
        train_values = _train_epoch(model, loader, optimizer, device, effective_config, observer, epoch)
        predictions = _quantile_predictions(model, validation_dataset.X, device)
        validation_skill, skill_by_site = _primary_site_mean_skill(
            validation_dataset, predictions, sites
        )
        observer.add_predictions(
            "validation", epoch, predictions, validation_dataset.Y, validation_dataset.mask
        )
        observer.add_primary_endpoint(
            epoch, validation_dataset, predictions, sites, validation_skill
        )
        validation_primary_mse, validation_primary_count = _pooled_primary_mse(
            validation_dataset, predictions
        )
        val_metrics, _, _ = _evaluate_predictions(
            validation_dataset,
            predictions,
            split="validation",
            seed=SEED,
            run_tag=run_id,
            sites=sites,
        )
        primary_rows = val_metrics.loc[
            (val_metrics["target_column"] == "spei_3")
            & val_metrics["lead"].isin(("T+3", "T+4"))
            & (val_metrics["site"] == "ALL")
        ]
        scheduler.step(validation_skill)
        is_best = stopper.update(validation_skill, epoch)
        history.append({
            "epoch": epoch,
            "seed": SEED,
            "train_total_objective": train_values["total"],
            "train_pinball_component": train_values["pinball"],
            "train_primary_mse_component": train_values["primary_mse"],
            "validation_primary_pooled_mse": validation_primary_mse,
            "validation_primary_pooled_mse_n": validation_primary_count,
            "validation_primary_site_mean_mse_skill": validation_skill,
            "validation_macro_site_spei3_t3_t4_rmse": float(primary_rows["rmse"].mean()),
            "validation_severe_recall": float(primary_rows["severe_or_extreme_recall"].mean()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "best_epoch_so_far": stopper.best_epoch,
            "selected_this_epoch": bool(is_best),
        })
        observer.finish_epoch(epoch)
        if is_best:
            selected_skill_by_site = skill_by_site
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, checkpoint_path)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} pinball={train_values['pinball']:.5f} "
                f"primary_mse={train_values['primary_mse']:.5f} "
                f"skill={validation_skill:+.5f}"
            )
        if stopper.should_stop:
            break
    if best_state is None or stopper.best_epoch is None:
        raise RuntimeError("No validation-selected alternative-objective checkpoint")

    pd.DataFrame(history).to_csv(result_dir / "objective_training_history.csv", index=False)
    selection = {
        "selected_epoch": stopper.best_epoch,
        "best_validation_primary_site_mean_mse_skill": stopper.best_score,
        "selection_rule": "unchanged Phase 2.2 equal-site SPEI-3 T+3/T+4 MSE skill versus persistence",
        "validation_skill_by_site_at_selected_epoch": selected_skill_by_site,
        "checkpoint_sha256": sha256(checkpoint_path),
    }
    metadata = {
        "experiment": "Phase 2.5.1B objective-alignment alternative",
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "profile": PROFILE_NAME,
        "dataset_sha256": dataset_hash,
        "normalizer_sha256": normalizer_hash,
        "protocol_version": DATASET_PROTOCOL_VERSION,
        "features": list(MODEL_FEATURES),
        "split_boundaries": SPLIT_BOUNDARIES,
        "window_counts": {"train": len(train_dataset), "validation": len(validation_dataset), "test_placeholder": len(datasets["test"])},
        "train_only_optimization": True,
        "validation_only_exploratory_selection_and_assessment": True,
        "test_split_windows_constructed": False,
        "test_split_accessed_or_evaluated": False,
        "objective": {
            "baseline": "sum of the unchanged three scale-wise masked extreme-weighted pinball losses",
            "alternative": "L_alt = L_pinball + 0.5 * MSE_primary",
            "mse_primary": "mean squared q50 error pooled over valid training examples and SPEI-3 leads T+3/T+4",
            "primary_mse_weight": PRIMARY_MSE_WEIGHT,
            "all_scale_lead_quantile_pinball_supervision_retained": True,
        },
        "effective_config": effective_config,
        "selected_epoch": stopper.best_epoch,
        "early_stop_epoch": history[-1]["epoch"],
        "selection": selection,
        "checkpoint_path": str(checkpoint_path),
        "source_hashes": {
            "runner": sha256(source_path),
            "phase2_2_runner": sha256(Path("experiments/run_phase2_2_neural.py")),
            "observer": sha256(Path("experiments/training_diagnostics.py")),
            "dataset_metadata": sha256(dataset_metadata),
            "config": sha256(CONFIG_PATH),
            "baseline_report": sha256(PHASE251_BASELINE / "baseline_experiment_report.md"),
        },
        "baseline_run_id": baseline_meta["run_id"],
        "phase2_2_run_id": phase22["run_id"],
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device)},
    }
    (result_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (result_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (result_dir / "effective_configuration.json").write_text(
        json.dumps(metadata["effective_config"], indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if sha256(dataset_path) != dataset_hash or sha256(NORMALIZER_PATH) != normalizer_hash:
        raise RuntimeError("Canonical data or normalizer changed during the run")
    if CONFIG_PATH.read_bytes() != config_bytes:
        raise RuntimeError("Configuration changed during the run")
    return {"run_id": run_id, "result_dir": str(result_dir), "model_dir": str(model_dir), "selection": selection}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
