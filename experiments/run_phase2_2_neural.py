"""Validation-selected Phase 2.2 DroughtCast quantile-model experiment.

The script keeps the protocol-10 data and model architecture fixed. A small,
predeclared hyperparameter screen is selected by validation SPEI-3 T+3/T+4
mean-per-site MSE skill against persistence. The test split is read only after
the checkpoint/configuration manifest is frozen.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr
from scipy.stats import t as student_t
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from torch.utils.data import DataLoader, TensorDataset

from data.loader import load_site_table, validate_dataset_protocol
from data.preprocessing import validate_normalizer_metadata
from data.protocol import (
    DATASET_PROTOCOL_VERSION,
    MODEL_FEATURES,
    SPLIT_BOUNDARIES,
)
from experiments.training_diagnostics import make_diagnostics
from models.forecaster import MultiScaleForecaster
from train import pinball_loss

TARGETS = ("spei_3", "spei_6", "spei_12")
TARGET_LABELS = {"spei_3": "SPEI-3", "spei_6": "SPEI-6", "spei_12": "SPEI-12"}
SCALE_INDEX = {name: index for index, name in enumerate(TARGETS)}
QUANTILES = (0.1, 0.5, 0.9)
CLASS_NAMES = (
    "extreme_drought",
    "severe_drought",
    "moderate_drought",
    "near_normal",
    "moderately_wet",
    "very_wet",
    "extremely_wet",
)
SEEDS = (42, 101, 2024, 2025, 2026)
BASELINE_MODELS = ("persistence", "ridge", "random_forest", "xgboost", "seasonal_climatology")
NORMALIZER_PATH = Path("data/normalizers/maharashtra_protocol10.json")
BASELINE_RUN = Path(
    "results/maharashtra/phase2/baselines/"
    "protocol10_seed42_e9dfa7ce76_ae290b4115_6f4f90ae"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def drought_categories(values: np.ndarray) -> np.ndarray:
    """Project seven-class boundaries; -1 denotes a missing target."""
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, -1, dtype=np.int8)
    valid = np.isfinite(values)
    result[valid & (values <= -2.0)] = 0
    result[valid & (values > -2.0) & (values <= -1.5)] = 1
    result[valid & (values > -1.5) & (values <= -1.0)] = 2
    result[valid & (values > -1.0) & (values < 1.0)] = 3
    result[valid & (values >= 1.0) & (values < 1.5)] = 4
    result[valid & (values >= 1.5) & (values < 2.0)] = 5
    result[valid & (values >= 2.0)] = 6
    return result


class ValidationEarlyStopper:
    """Track improvements in a validation-only score and patience."""

    def __init__(self, patience: int, min_delta: float = 1e-5):
        if patience < 1:
            raise ValueError("patience must be positive")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_score = -np.inf
        self.best_epoch: int | None = None
        self.bad_epochs = 0

    def update(self, score: float, epoch: int) -> bool:
        if not np.isfinite(score):
            raise ValueError("Validation selection score must be finite")
        if score > self.best_score + self.min_delta:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def _model_config(base_config: dict, profile: dict) -> dict:
    config = copy.deepcopy(base_config)
    # The model's strict schema accepts only model-consumed data settings;
    # this dataset-builder calibration field stays in the recorded source
    # config but is not passed into the architecture constructor.
    config["data"].pop("spei_calibration_start_year", None)
    model = config["model"]
    training = config["training"]
    model.update(
        {
            "encoder": "lstm",
            "attention": "shared",
            "heads": "direct",
            "hidden_dim_1": profile["hidden_dim_1"],
            "hidden_dim_2": profile["hidden_dim_2"],
            "dropout": profile["dropout"],
            "use_cascaded_decoder": False,
            "quantiles": list(QUANTILES),
        }
    )
    # The implementation reads `variable_gate`. Drop the legacy ignored alias
    # so the serialized experiment config reflects the effective model.
    model.pop("use_variable_gate", None)
    model["variable_gate"] = False
    training.update(
        {
            "learning_rate": profile["learning_rate"],
            "weight_decay": profile["weight_decay"],
            "batch_size": profile["batch_size"],
            "loss": "pinball",
            "task_weighting": "fixed",
        }
    )
    return config


def _quantile_predictions(model: MultiScaleForecaster, x: torch.Tensor, device) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        output, _ = model(x.to(device=device, dtype=torch.float32))
    result = {key: value.detach().cpu().numpy() for key, value in output.items()}
    expected = (len(x), 4, 3)
    for scale in TARGETS:
        if result[scale].shape != expected or not np.isfinite(result[scale]).all():
            raise RuntimeError(f"Invalid {scale} quantile output: {result[scale].shape}")
        if np.any(result[scale][..., 0] > result[scale][..., 1]) or np.any(
            result[scale][..., 1] > result[scale][..., 2]
        ):
            raise RuntimeError(f"Quantile crossing detected in {scale}")
    return result


def _primary_site_mean_skill(
    dataset,
    predictions: dict[str, np.ndarray],
    sites: list[str],
) -> tuple[float, dict[str, float]]:
    """Primary endpoint: site-macro MSE skill vs persistence, SPEI-3 T+3/T+4."""
    target = dataset.Y.numpy()[:, 0, 2:4]
    mask = dataset.mask.numpy()[:, 0, 2:4]
    median = predictions["spei_3"][:, 2:4, 1]
    persistence = np.asarray(
        [
            float(metadata["last_observed_spei"]["spei_3"])
            for metadata in dataset.metadata
        ],
        dtype=float,
    )[:, None]
    site_ids = np.asarray([str(metadata["site"]) for metadata in dataset.metadata])
    scores = {}
    for site in sites:
        site_mask = (site_ids == site)[:, None] & mask & np.isfinite(target)
        if not site_mask.any():
            raise RuntimeError(f"Primary endpoint has no valid observations for site={site}")
        observed = target[site_mask]
        model_values = median[site_mask]
        persistence_values = np.broadcast_to(persistence, target.shape)[site_mask]
        baseline_mse = float(np.mean((observed - persistence_values) ** 2))
        if baseline_mse <= 0:
            raise RuntimeError(f"Persistence MSE is zero for validation site={site}")
        scores[site] = 1.0 - float(np.mean((observed - model_values) ** 2)) / baseline_mse
    return float(np.mean(list(scores.values()))), scores


def _train_epoch(model, loader, optimizer, device, config, diagnostics=None, epoch=None) -> float:
    model.train()
    total, count = 0.0, 0
    for x, y, mask in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device)
        optimizer.zero_grad(set_to_none=True)
        predictions, _ = model(x)
        losses = [
            pinball_loss(
                predictions[scale],
                y[:, scale_index],
                mask[:, scale_index],
                QUANTILES,
                float(config["training"].get("extreme_weight", 1.0)),
                float(config["training"].get("extreme_threshold", -1.0)),
            )
            for scale_index, scale in enumerate(TARGETS)
        ]
        loss = torch.stack(losses).sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training pinball loss")
        if diagnostics is not None:
            diagnostics.add_predictions("train", epoch, predictions, y, mask)
        loss.backward()
        if diagnostics is not None:
            parameter_snapshot = diagnostics.before_optimizer_step(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip", 1.0)))
        optimizer.step()
        if diagnostics is not None:
            diagnostics.after_optimizer_step(model, parameter_snapshot)
        total += float(loss.detach().cpu()) * len(x)
        count += len(x)
    return total / max(count, 1)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.asarray(y_true, dtype=float)[valid]
    y_pred = np.asarray(y_pred, dtype=float)[valid]
    n = len(y_true)
    if n == 0:
        return {"n_valid": 0, "mae": None, "rmse": None, "r2": None, "pearson_r": None}
    correlation = None
    if n >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        correlation = float(pearsonr(y_true, y_pred).statistic)
    return {
        "n_valid": n,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if n >= 2 and np.std(y_true) > 0 else None,
        "pearson_r": correlation,
    }


def _binary_scores(actual: np.ndarray, predicted: np.ndarray) -> dict:
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _evaluate_predictions(
    dataset,
    quantile_predictions: dict[str, np.ndarray],
    *,
    split: str,
    seed: int,
    run_tag: str,
    sites: list[str],
) -> tuple[pd.DataFrame, dict, list[dict]]:
    y = dataset.Y.numpy()
    mask = dataset.mask.numpy().astype(bool)
    site_ids = np.asarray([str(row["site"]) for row in dataset.metadata])
    metrics, confusion, class_counts = [], {}, []
    for scale_index, target_name in enumerate(TARGETS):
        q_predictions = quantile_predictions[target_name]
        for lead in range(4):
            target = y[:, scale_index, lead]
            valid_target = mask[:, scale_index, lead] & np.isfinite(target)
            q_values = q_predictions[:, lead, :]
            median = q_values[:, 1]
            for site in [*sites, "ALL"]:
                selected = np.ones(len(site_ids), dtype=bool) if site == "ALL" else site_ids == site
                valid = selected & valid_target & np.isfinite(median)
                yv, pv, qv = target[valid], median[valid], q_values[valid]
                yc = drought_categories(yv)
                pc = drought_categories(pv)
                class_metrics = {
                    "category_accuracy": float(accuracy_score(yc, pc)) if len(yc) else None,
                    "macro_f1": float(f1_score(yc, pc, labels=np.arange(7), average="macro", zero_division=0)) if len(yc) else None,
                    "weighted_f1": float(f1_score(yc, pc, labels=np.arange(7), average="weighted", zero_division=0)) if len(yc) else None,
                }
                severe_actual, severe_pred = yc == 1, pc == 1
                severe = _binary_scores(severe_actual, severe_pred)
                severe_or_extreme = _binary_scores(yv <= -1.5, pv <= -1.5)
                regression = _regression_metrics(yv, pv)
                pinball = {}
                for q_index, q in enumerate(QUANTILES):
                    error = yv - qv[:, q_index]
                    pinball[f"pinball_q{int(q * 100):02d}"] = (
                        float(np.mean(np.maximum(q * error, (q - 1) * error)))
                        if len(error)
                        else None
                    )
                interval_covered = (yv >= qv[:, 0]) & (yv <= qv[:, 2]) if len(yv) else np.array([], dtype=bool)
                interval = {
                    "q10_q90_coverage": float(interval_covered.mean()) if len(interval_covered) else None,
                    "q10_q90_width": float(np.mean(qv[:, 2] - qv[:, 0])) if len(qv) else None,
                }
                metrics.append(
                    {
                        "split": split,
                        "run_tag": run_tag,
                        "seed": seed,
                        "model": "droughtcast_quantile_lstm_shared_attention",
                        "site": site,
                        "target": TARGET_LABELS[target_name],
                        "target_column": target_name,
                        "lead": f"T+{lead + 1}",
                        "n_total_windows": int(selected.sum()),
                        "n_valid": regression["n_valid"],
                        "n_excluded_missing_target_or_prediction": int(selected.sum() - regression["n_valid"]),
                        **regression,
                        **class_metrics,
                        "severe_drought_threshold": "-2.0 < SPEI <= -1.5",
                        "severe_drought_precision": severe["precision"],
                        "severe_drought_recall": severe["recall"],
                        "severe_drought_f1": severe["f1"],
                        "severe_drought_tp": severe["tp"],
                        "severe_drought_fp": severe["fp"],
                        "severe_drought_fn": severe["fn"],
                        "severe_or_extreme_threshold": "SPEI <= -1.5",
                        "severe_or_extreme_precision": severe_or_extreme["precision"],
                        "severe_or_extreme_recall": severe_or_extreme["recall"],
                        "severe_or_extreme_f1": severe_or_extreme["f1"],
                        **pinball,
                        **interval,
                    }
                )
                matrix = confusion_matrix(yc, pc, labels=np.arange(7))
                per_class_p, per_class_r, per_class_f, per_class_n = precision_recall_fscore_support(
                    yc, pc, labels=np.arange(7), zero_division=0
                )
                confusion[f"{split}|{run_tag}|{target_name}|T+{lead + 1}|{site}"] = {
                    "labels": list(CLASS_NAMES),
                    "matrix_true_rows_predicted_columns": matrix.astype(int).tolist(),
                    "per_class": {
                        CLASS_NAMES[index]: {
                            "precision": float(per_class_p[index]),
                            "recall": float(per_class_r[index]),
                            "f1": float(per_class_f[index]),
                            "support": per_class_n[index].item(),
                        }
                        for index in range(7)
                    },
                    "severe_class": severe,
                    "severe_or_extreme_event": severe_or_extreme,
                    "n_valid": len(yv),
                }
                counts = np.bincount(yc[yc >= 0], minlength=7)
                for category_index, category_name in enumerate(CLASS_NAMES):
                    class_counts.append(
                        {
                            "split": split,
                            "run_tag": run_tag,
                            "seed": seed,
                            "site": site,
                            "target": TARGET_LABELS[target_name],
                            "target_column": target_name,
                            "lead": f"T+{lead + 1}",
                            "category_code": category_index,
                            "category": category_name,
                            "count": counts[category_index].item(),
                            "valid_target_count": len(yc),
                            "missing_target_count": int(selected.sum() - valid_target[selected].sum()),
                        }
                    )
    return pd.DataFrame(metrics), confusion, class_counts


def _validation_outputs(model, dataset, device) -> dict[str, np.ndarray]:
    return _quantile_predictions(model, dataset.X, device)


def _fit_profile(
    base_config: dict,
    profile_name: str,
    profile: dict,
    train_dataset,
    validation_dataset,
    sites: list[str],
    device: torch.device,
    seed: int,
    checkpoint_path: Path,
    diagnostics=None,
) -> tuple[MultiScaleForecaster, dict, list[dict]]:
    config = _model_config(base_config, profile)
    seed_everything(seed)
    model = MultiScaleForecaster(config).to(device)
    train_data = TensorDataset(train_dataset.X, train_dataset.Y, train_dataset.mask)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
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
        optimizer,
        mode="max",
        patience=int(base_config["training"].get("lr_patience", 5)),
        factor=0.5,
    )
    stopper = ValidationEarlyStopper(
        int(base_config["training"].get("early_stopping_patience", 12))
    )
    history: list[dict] = []
    best_state = None
    max_epochs = int(base_config["training"].get("epochs", 60))
    for epoch in range(1, max_epochs + 1):
        train_loss = _train_epoch(
            model, train_loader, optimizer, device, config, diagnostics, epoch
        )
        val_predictions = _validation_outputs(model, validation_dataset, device)
        val_skill, _ = _primary_site_mean_skill(
            validation_dataset, val_predictions, sites
        )
        val_metrics, _, _ = _evaluate_predictions(
            validation_dataset,
            val_predictions,
            split="validation",
            seed=seed,
            run_tag=profile_name,
            sites=sites,
        )
        if diagnostics is not None:
            diagnostics.add_predictions(
                "validation", epoch, val_predictions,
                validation_dataset.Y, validation_dataset.mask,
            )
            diagnostics.add_primary_endpoint(
                epoch, validation_dataset, val_predictions, sites, val_skill
            )
        primary_rows = val_metrics.loc[
            (val_metrics["target_column"] == "spei_3")
            & val_metrics["lead"].isin(("T+3", "T+4"))
            & (val_metrics["site"] == "ALL")
        ]
        macro_severe_recall = float(primary_rows["severe_drought_recall"].mean())
        macro_site_rmse = float(
            val_metrics.loc[
                (val_metrics["target_column"] == "spei_3")
                & val_metrics["lead"].isin(("T+3", "T+4"))
                & (val_metrics["site"] != "ALL"),
                "rmse",
            ].mean()
        )
        is_best = stopper.update(val_skill, epoch)
        scheduler.step(val_skill)
        history.append(
            {
                "profile": profile_name,
                "seed": seed,
                "epoch": epoch,
                "train_pinball_loss": train_loss,
                "validation_primary_site_mean_mse_skill": val_skill,
                "validation_macro_site_spei3_t3_t4_rmse": macro_site_rmse,
                "validation_severe_category_recall_spei3_t3_t4": macro_severe_recall,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "best_epoch_so_far": stopper.best_epoch,
            }
        )
        if is_best:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save(best_state, checkpoint_path)
        if diagnostics is not None:
            diagnostics.finish_epoch(epoch)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"profile={profile_name} seed={seed} epoch={epoch:03d} "
                f"train_pinball={train_loss:.5f} val_skill={val_skill:+.5f} "
                f"val_severe_recall={macro_severe_recall:.4f}"
            )
        if stopper.should_stop:
            print(f"profile={profile_name} seed={seed}: early stop at epoch={epoch}")
            break
    if best_state is None or stopper.best_epoch is None:
        raise RuntimeError(f"No validation-selected checkpoint for profile {profile_name}")
    model.load_state_dict(best_state)
    model.eval()
    info = {
        "profile": profile_name,
        "seed": seed,
        "selected_epoch": stopper.best_epoch,
        "best_validation_primary_site_mean_mse_skill": stopper.best_score,
        "validation_skill_by_site": _primary_site_mean_skill(
            validation_dataset, _validation_outputs(model, validation_dataset, device), sites
        )[1],
        "profile_parameters": profile,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "selection_split": "validation",
        "selection_rule": "maximum equal-site mean of per-site SPEI-3 MSE skill vs persistence across T+3 and T+4",
    }
    return model, info, history


def _package_versions() -> dict:
    result = {"python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "pyyaml"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _assert_baseline_contract(baseline_run: Path, dataset_hash: str, normalizer_hash: str) -> tuple[dict, dict]:
    metadata_path = baseline_run / "baseline_experiment_metadata.json"
    leakage_path = baseline_run / "baseline_leakage_audit.json"
    metrics_path = baseline_run / "baseline_metrics.csv"
    for path in (metadata_path, leakage_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Phase 2.1 artifact missing: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    if metadata.get("dataset_sha256") != dataset_hash:
        raise RuntimeError("Phase 2.1 baseline dataset fingerprint differs")
    if metadata.get("normalizer_sha256") != normalizer_hash:
        raise RuntimeError("Phase 2.1 baseline normalizer hash differs")
    if metadata.get("protocol_version") != DATASET_PROTOCOL_VERSION:
        raise RuntimeError("Phase 2.1 baseline protocol differs")
    if metadata.get("features") != MODEL_FEATURES:
        raise RuntimeError("Phase 2.1 baseline feature order differs")
    if metadata.get("split_boundaries") != SPLIT_BOUNDARIES:
        raise RuntimeError("Phase 2.1 baseline split boundaries differ")
    required_flags = (
        "no_cross_split_window_detected",
        "no_future_targets_in_input_windows",
        "no_future_climate_index_dates",
        "target_derived_features_are_same_month_only",
        "no_over_age_indices_used_as_finite_inputs",
    )
    if any(not leakage.get(key, False) for key in required_flags):
        raise RuntimeError("Phase 2.1 leakage audit is incomplete or failed")
    return metadata, leakage


def _verify_normalizer(metadata, expected_hash, dataset_hash, config):
    if sha256(NORMALIZER_PATH) != expected_hash:
        raise RuntimeError("Protocol-10 normalizer hash changed")
    validate_normalizer_metadata(
        metadata,
        list(MODEL_FEATURES),
        expected_dataset_sha256=dataset_hash,
        expected_protocol_version=DATASET_PROTOCOL_VERSION,
        expected_split_boundaries=SPLIT_BOUNDARIES,
        expected_seq_len=int(config["data"]["seq_len"]),
        expected_forecast_steps=int(config["data"]["forecast_steps"]),
    )


def _verify_baseline_sample_counts(baseline_run: Path, dataset, sites: list[str]) -> None:
    baseline = pd.read_csv(baseline_run / "baseline_metrics.csv")
    baseline = baseline.loc[
        (baseline["feature_variant"] == "all_features")
        & baseline["model"].isin(BASELINE_MODELS)
    ]
    expected = {}
    target_masks = dataset.mask.numpy()
    site_ids = np.asarray([str(item["site"]) for item in dataset.metadata])
    for scale_index, target in enumerate(TARGETS):
        for lead in range(4):
            for site in [*sites, "ALL"]:
                selected = np.ones(len(site_ids), dtype=bool) if site == "ALL" else site_ids == site
                expected[(target, f"T+{lead + 1}", site)] = int(
                    target_masks[selected, scale_index, lead].sum()
                )
    observed = baseline.groupby(["target_column", "lead", "site"])["n_valid"].first()
    for key, expected_count in expected.items():
        if key not in observed or int(observed.loc[key]) != expected_count:
            raise RuntimeError(
                f"Baseline/test target alignment count mismatch for {key}: "
                f"expected={expected_count}, baseline={observed.get(key)}"
            )


def _class_distribution(dataset, split: str, sites: list[str], seed: int | None = None) -> list[dict]:
    y, mask = dataset.Y.numpy(), dataset.mask.numpy().astype(bool)
    site_ids = np.asarray([str(item["site"]) for item in dataset.metadata])
    records = []
    for scale_index, target_name in enumerate(TARGETS):
        for lead in range(y.shape[2]):
            for site in [*sites, "ALL"]:
                selected = np.ones(len(site_ids), dtype=bool) if site == "ALL" else site_ids == site
                valid = selected & mask[:, scale_index, lead] & np.isfinite(y[:, scale_index, lead])
                cats = drought_categories(y[valid, scale_index, lead])
                counts = np.bincount(cats[cats >= 0], minlength=7)
                for index, name in enumerate(CLASS_NAMES):
                    records.append(
                        {
                            "split": split,
                            "seed": seed,
                            "site": site,
                            "target": TARGET_LABELS[target_name],
                            "target_column": target_name,
                            "lead": f"T+{lead + 1}",
                            "category_code": index,
                            "category": name,
                            "count": counts[index].item(),
                            "valid_target_count": len(cats),
                            "missing_target_count": int(selected.sum() - valid.sum()),
                        }
                    )
    return records


def _make_baseline_comparison(run_dir: Path, baseline_run: Path, test_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_run / "baseline_metrics.csv")
    baseline = baseline.loc[
        (baseline["feature_variant"] == "all_features")
        & baseline["model"].isin(BASELINE_MODELS)
        & (baseline["site"] != "ALL"),
    ].copy()
    neural = test_metrics.copy()
    grouped = neural.groupby(["target_column", "lead", "site"], as_index=False).agg(
        neural_seed42_rmse=("rmse", "first"),
        neural_mean_seed_rmse=("rmse", "mean"),
        neural_std_seed_rmse=("rmse", "std"),
        neural_mean_seed_mae=("mae", "mean"),
        neural_mean_seed_macro_f1=("macro_f1", "mean"),
        neural_mean_seed_severe_recall=("severe_drought_recall", "mean"),
        neural_mean_seed_severe_or_extreme_recall=("severe_or_extreme_recall", "mean"),
        neural_n_valid=("n_valid", "first"),
        neural_seed_count=("seed", "nunique"),
    )
    critical_value = float(student_t.ppf(0.975, df=max(len(neural["seed"].unique()) - 1, 1)))
    standard_error = grouped["neural_std_seed_rmse"] / np.sqrt(grouped["neural_seed_count"])
    grouped["neural_rmse_seed_ci95_lower"] = grouped["neural_mean_seed_rmse"] - critical_value * standard_error
    grouped["neural_rmse_seed_ci95_upper"] = grouped["neural_mean_seed_rmse"] + critical_value * standard_error
    # seed 42 is the locked primary checkpoint; remaining rows summarize seed spread.
    seed42 = neural.loc[neural["seed"] == 42, ["target_column", "lead", "site", "rmse", "mae", "macro_f1", "severe_drought_recall", "severe_or_extreme_recall", "n_valid"]].rename(
        columns={
            "rmse": "neural_seed42_rmse",
            "mae": "neural_seed42_mae",
            "macro_f1": "neural_seed42_macro_f1",
            "severe_drought_recall": "neural_seed42_severe_recall",
            "severe_or_extreme_recall": "neural_seed42_severe_or_extreme_recall",
            "n_valid": "neural_n_valid",
        }
    )
    grouped = grouped.drop(columns=["neural_seed42_rmse", "neural_n_valid"]).merge(
        seed42, on=["target_column", "lead", "site"], validate="one_to_one"
    )
    baseline = baseline.merge(grouped, on=["target_column", "lead", "site"], validate="many_to_one")
    if not (baseline["n_valid"].to_numpy() == baseline["neural_n_valid"].to_numpy()).all():
        raise RuntimeError("Neural and baseline test metric sample counts differ")
    baseline["delta_seed42_rmse_vs_baseline"] = baseline["neural_seed42_rmse"] - baseline["rmse"]
    baseline["delta_seed_mean_rmse_vs_baseline"] = baseline["neural_mean_seed_rmse"] - baseline["rmse"]
    baseline["test_set_consumed"] = True
    baseline.to_csv(run_dir / "neural_baseline_comparison.csv", index=False)
    return baseline


def _report(run_dir: Path, metadata: dict, validation_metrics: pd.DataFrame, test_metrics: pd.DataFrame, comparison: pd.DataFrame) -> None:
    all_val = validation_metrics.loc[validation_metrics["site"] == "ALL"]
    val_selected = all_val.loc[all_val["run_tag"] == "selected_multiseed"]
    all_test = test_metrics.loc[test_metrics["site"] == "ALL"]
    baseline_all = comparison.loc[comparison["site"] == "ALL"] if "ALL" in comparison["site"].values else pd.DataFrame()
    if baseline_all.empty:
        baseline_metrics = pd.read_csv(BASELINE_RUN / "baseline_metrics.csv")
        baseline_all = baseline_metrics.loc[
            (baseline_metrics["feature_variant"] == "all_features")
            & baseline_metrics["model"].isin(BASELINE_MODELS)
            & (baseline_metrics["site"] == "ALL")
        ]
    lines = [
        "# Phase 2.2 — Proposed DroughtCast Neural Model",
        "",
        "**Decision: COMPLETE WITH LIMITATIONS**",
        "",
        "## Executive decision",
        "",
        "A validation-selected, protocol-10 quantile LSTM model was trained using the existing DroughtCast architecture. Its locked comparison uses the Phase 2.1 benchmark period, whose test results had already been consumed for baseline evaluation. This is a final comparative analysis on that period, not an untouched independent test. The neural model's test results were not used to revise the model.",
        "",
        "## Architecture implemented",
        "",
        f"The existing code is the proposed model; no architecture rewrite was needed. It uses a shared LSTM({metadata['selected_parameters']['hidden_dim_1']}) -> dropout({metadata['selected_parameters']['dropout']}) -> LSTM({metadata['selected_parameters']['hidden_dim_2']}) encoder, shared additive temporal attention, and three direct scale-specific quantile decoders. Each scale emits `[batch, 4 leads, 3 quantiles]`; keys map to SPEI-3, SPEI-6 and SPEI-12, while the final axis maps q10, q50, q90. q50 tensors are stacked as `[batch, 3 scales, 4 leads]`. The configured cascaded decoder is disabled for quantile outputs by the existing implementation. The optional variable gate is disabled, consistent with the model's actual configuration key.",
        "",
        "## Data and preprocessing verification",
        "",
        f"- Dataset SHA-256: `{metadata['dataset_sha256']}`.",
        f"- Normalizer SHA-256: `{metadata['normalizer_sha256']}` (matches Phase 2.1).",
        f"- Protocol: {metadata['protocol_version']}; 12 configured sites; 15 features; six input months; four leads.",
        f"- Window counts: train {metadata['window_counts']['train']}, validation {metadata['window_counts']['val']}, test {metadata['window_counts']['test']}.",
        "- Windows: train inputs 1990-01–2010-08, targets 1990-07–2010-12; validation inputs 2012-01–2015-08, targets 2012-07–2015-12; test inputs 2017-01–2025-08, targets 2017-07–2025-12.",
        "- The train-only per-site normalizer is reused; no scaler or imputer was fit on validation or test. Missing features retain the training-mean imputation policy.",
        "",
        "## Training and validation-only selection",
        "",
        f"The small predeclared profile screen selected `{metadata['selected_profile']}`. Selection and early stopping used the configured primary endpoint: maximum equal-site mean SPEI-3 MSE skill versus persistence across T+3 and T+4. Each candidate used validation only for checkpoint selection; test labels were not read until after the evaluation manifest was frozen.",
        "",
        "| Seed/profile | Selected epoch | Validation site-mean skill vs persistence |",
        "|---|---:|---:|",
    ]
    for record in metadata["candidate_selection"]:
        lines.append(
            f"| {record['profile']} / seed {record['seed']} | {record['selected_epoch']} | {record['best_validation_primary_site_mean_mse_skill']:+.4f} |"
        )
    lines += [
        "",
        "Training used pinball loss for q10/q50/q90, fixed seed, AdamW, validation-based ReduceLROnPlateau, gradient clipping, and early stopping. The selected profile and complete training curves are in the machine-readable artifacts.",
        "",
        "## Validation results",
        "",
        "`neural_metrics_validation.csv` reports MAE, RMSE, R², Pearson correlation, seven-category accuracy/macro/weighted F1, exact severe-category scores, severe-or-extreme event scores, per-class confusion details, quantile pinball loss, and q10–q90 coverage/width for each scale, lead, site and seed. This is validation performance, not a test estimate.",
        "",
        "| Target | Lead | q50 RMSE | MAE | R² | Macro F1 | Severe recall | n |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        for lead in range(1, 5):
            rows = val_selected.loc[
                (val_selected["target_column"] == target)
                & (val_selected["lead"] == f"T+{lead}")
            ]
            if rows.empty:
                continue
            lines.append(
                f"| {TARGET_LABELS[target]} | T+{lead} | {rows['rmse'].mean():.4f} | {rows['mae'].mean():.4f} | {rows['r2'].mean():.4f} | {rows['macro_f1'].mean():.4f} | {rows['severe_drought_recall'].mean():.4f} | {int(rows['n_valid'].min())} |"
            )
    lines += [
        "",
        "## Locked test and Phase 2.1 comparison",
        "",
        "The checkpoint list, configuration, dataset and normalizer hashes, evaluation code hash, metric definitions, and baseline artifact hashes were written to the freeze manifest before test prediction. All five selected-profile seeds were then evaluated once. Test outputs are retained for final comparison only; no test-driven changes followed.",
        "",
        "| Target | Lead | Neural seed-42 RMSE | Neural five-seed mean RMSE | Persistence RMSE | Ridge RMSE | RF RMSE | XGBoost RMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        for lead in range(1, 5):
            neural_rows = all_test.loc[
                (all_test["target_column"] == target)
                & (all_test["lead"] == f"T+{lead}")
            ]
            if neural_rows.empty:
                continue
            by_model = baseline_all.loc[
                (baseline_all["target_column"] == target)
                & (baseline_all["lead"] == f"T+{lead}")
            ].groupby("model")["rmse"].first().to_dict()
            selected_mean = float(neural_rows["rmse"].mean())
            lines.append(
                f"| {TARGET_LABELS[target]} | T+{lead} | {neural_rows.loc[neural_rows.seed.eq(42), 'rmse'].iloc[0]:.4f} | {selected_mean:.4f} | {by_model.get('persistence', np.nan):.4f} | {by_model.get('ridge', np.nan):.4f} | {by_model.get('random_forest', np.nan):.4f} | {by_model.get('xgboost', np.nan):.4f} |"
            )
    lines += [
        "",
        "The full comparison includes site-level rows and MAE, macro F1, severe-category recall, severe-or-extreme recall, seed mean/standard deviation and a 95% t interval across the five training seeds. Those intervals describe seed variation only, not uncertainty across climate periods. Sample counts are checked against the frozen baseline results. Baseline test predictions were not regenerated. The benchmark test period is reused and therefore is not a fresh independent confirmation.",
        "",
        "## Severe drought and site-level performance",
        "",
        "The exact severe category is -2.0 < SPEI ≤ -1.5; extreme drought is SPEI ≤ -2.0. A separate binary severe-or-extreme event uses SPEI ≤ -1.5. Read `neural_metrics_test.csv`, `neural_per_site_metrics.csv`, and `neural_confusion_matrices.json` for each scale/lead/site. Report weak or zero-recall results without suppression.",
        "",
        "## Leakage audit",
        "",
        "The baseline window audit was revalidated before training: all input/target dates stayed inside one split, target dates did not enter their own input windows, no future source/effective climate-index month appeared, observed SPEI feature columns matched same-month historical targets, and train/validation/test target periods were disjoint. The exact verified normalizer hash was used. Gradient updates used training tensors only; validation drove profile/checkpoint selection; the test split was not read until freeze.",
        "",
        "## Reproducibility",
        "",
        f"Run ID: `{metadata['run_id']}`. Python/PyTorch: {metadata['environment']['packages'].get('python')} / {metadata['environment']['packages'].get('torch')}; device: {metadata['device']}; CUDA: {metadata['environment']['packages'].get('cuda')}. Random seeds: {', '.join(map(str, metadata['seeds']))}. Dataset, normalizer, run script, evaluation script and baseline hashes are recorded in `neural_experiment_metadata.json`.",
        "",
        "## Limitations and next steps",
        "",
        "This is a single protocol/data period and 12 represented sites; it does not establish spatial transfer. Locally calculated empirical SPEI has the Phase 1 target-validity limitations, and the baseline period was already used. Five-seed variability reflects initialization/training variation, not independent climate periods. No publication-readiness claim is made. A paper study should preserve the target caveats, report site and drought-event weaknesses, and seek an untouched temporal evaluation period or external validation.",
        "",
        "## Verification",
        "",
        f"Focused tests: {metadata['verification']['focused_tests']['summary']}. Full pytest: {metadata['verification']['full_pytest']['summary']}. Ruff: {metadata['verification']['ruff']['summary']}. Neural model training was authorized in this phase; only the proposed DroughtCast model was trained. No canonical data, target, feature or split changes were made.",
        "",
        "**Decision: COMPLETE WITH LIMITATIONS.**",
    ]
    (run_dir / "neural_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: Path,
    result_root: Path,
    model_root: Path,
    baseline_run: Path,
    seeds=SEEDS,
    diagnostics_dir: Path | None = None,
) -> dict:
    config_bytes_before = Path(config_path).read_bytes()
    config = yaml.safe_load(config_bytes_before)
    data_cfg = config["data"]
    dataset_path = Path(data_cfg["path"])
    dataset_hash = sha256(dataset_path)
    normalizer_hash = sha256(NORMALIZER_PATH)
    if dataset_hash != json.loads((baseline_run / "baseline_experiment_metadata.json").read_text())["dataset_sha256"]:
        raise RuntimeError("Current dataset differs from the Phase 2.1 benchmark dataset")
    baseline_meta, baseline_leakage = _assert_baseline_contract(
        baseline_run, dataset_hash, normalizer_hash
    )
    if tuple(seeds) != SEEDS:
        raise ValueError(f"The predeclared Phase 2.2 seeds are fixed as {SEEDS}")
    normalizer_metadata = json.loads(NORMALIZER_PATH.read_text(encoding="utf-8"))
    _verify_normalizer(normalizer_metadata, normalizer_hash, dataset_hash, config)
    if data_cfg["features"] != MODEL_FEATURES or int(config["model"]["input_dim"]) != 15:
        raise RuntimeError("Configured model feature contract is not protocol-10 exact")
    if int(data_cfg["seq_len"]) != 6 or int(data_cfg["forecast_steps"]) != 4:
        raise RuntimeError("Configured window dimensions differ from the Phase 2.1 contract")
    if data_cfg["split_boundaries"] != SPLIT_BOUNDARIES:
        raise RuntimeError("Configured split boundaries differ from protocol 10")
    validate_dataset_protocol(
        dataset_path.parent / "dataset_metadata.json",
        data_cfg.get("target_source", "local"),
        data_cfg["features"],
    )
    dataset_protocol = json.loads((dataset_path.parent / "dataset_metadata.json").read_text())
    if dataset_protocol.get("schema_version") != DATASET_PROTOCOL_VERSION:
        raise RuntimeError("Dataset schema version differs from protocol 10")

    verification_path = Path(__file__).with_name("phase2_2_verification.json")
    if not verification_path.is_file():
        raise RuntimeError("Run and record focused/full tests and Ruff before training")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if not all(
        verification.get(key, {}).get("passed", False)
        for key in ("focused_tests", "full_pytest", "ruff")
    ):
        raise RuntimeError("Phase 2.2 verification record does not show passing checks")

    script_hash = sha256(Path(__file__))
    run_id = (
        f"protocol{DATASET_PROTOCOL_VERSION}_seed42_"
        f"{dataset_hash[:10]}_{normalizer_hash[:10]}_{script_hash[:8]}"
    )
    result_dir, checkpoint_dir = result_root / run_id, model_root / run_id
    if result_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to overwrite neural run {run_id}")
    diagnostics_path = Path(diagnostics_dir) if diagnostics_dir is not None else None
    if diagnostics_path is not None and diagnostics_path.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostics directory {diagnostics_path}")
    result_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    if diagnostics_path is not None:
        diagnostics_path.mkdir(parents=True)
        (diagnostics_path / "diagnostic_run_metadata.json").write_text(
            json.dumps({
                "run_id": run_id,
                "diagnostics_enabled": True,
                "dataset_sha256": dataset_hash,
                "normalizer_sha256": normalizer_hash,
                "source_config_sha256": sha256(config_path),
                "runner_sha256": script_hash,
                "split_used_for_training": "train",
                "split_used_for_selection": "validation",
                "test_instrumentation": False,
            }, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    # Load windows via the verified common loader. It fits no new normalizer:
    # the supplied versioned artifact is checked against train-only statistics.
    data = load_site_table(
        dataset_path,
        features=list(MODEL_FEATURES),
        seq_len=6,
        forecast_steps=4,
        metadata_path=NORMALIZER_PATH,
        site_filter=list(data_cfg["site_filter"]),
        site_specs=config.get("sites"),
        split_config=SPLIT_BOUNDARIES,
        include_splits=("train", "val"),
    )
    train_data, val_data = data["train"], data["val"]
    sites = sorted(data_cfg["site_filter"])
    development_window_counts = {key: len(data[key]) for key in ("train", "val")}
    if development_window_counts != {"train": 2916, "val": 468}:
        raise RuntimeError(
            f"Current loader window counts differ from verified benchmark: {development_window_counts}"
        )
    missingness = baseline_leakage.get("input_missingness_by_split", {})
    expected_missingness = {
        "train": (2916, 132, 900),
        "val": (468, 0, 0),
        "test": (1188, 0, 0),
    }
    for split_name, expected in expected_missingness.items():
        observed = missingness.get(split_name, {})
        actual = (
            observed.get("windows"),
            observed.get("windows_with_missing_inputs"),
            observed.get("missing_feature_cells"),
        )
        if actual != expected:
            raise RuntimeError(
                f"Phase 2.1 missing-input audit changed for {split_name}: {actual} != {expected}"
            )
    window_counts = {
        **development_window_counts,
        "test": int(baseline_meta["window_counts"]["test"]),
    }
    for split_name, dataset in (("train", train_data), ("val", val_data)):
        if not torch.isfinite(dataset.X).all():
            raise RuntimeError(f"Non-finite model input in {split_name}")
    if config_bytes_before != Path(config_path).read_bytes():
        raise RuntimeError("Source config changed during Phase 2.2 preflight")
    if sha256(dataset_path) != dataset_hash or sha256(NORMALIZER_PATH) != normalizer_hash:
        raise RuntimeError("Canonical dataset or normalizer changed during preflight")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The current environment has CPU-only PyTorch; keep a small deterministic
    # thread pool for its short sequences and modest batches.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    profiles = {
        "configured_base": {
            "hidden_dim_1": int(config["model"].get("hidden_dim_1", 64)),
            "hidden_dim_2": int(config["model"].get("hidden_dim_2", 32)),
            "dropout": float(config["model"].get("dropout", 0.2)),
            "learning_rate": float(config["training"].get("learning_rate", 1e-3)),
            "weight_decay": float(config["training"].get("weight_decay", 1e-4)),
            "batch_size": int(config["training"].get("batch_size", 32)),
        },
        "compact_regularized": {
            "hidden_dim_1": 32,
            "hidden_dim_2": 16,
            "dropout": 0.3,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 32,
        },
        "regularized_batch64": {
            "hidden_dim_1": 64,
            "hidden_dim_2": 32,
            "dropout": 0.3,
            "learning_rate": 5e-4,
            "weight_decay": 5e-4,
            "batch_size": 64,
        },
    }
    candidate_selection, history_records, validation_records = [], [], []
    candidate_paths = {}
    candidate_confusions = {}
    candidate_class_counts = []
    for profile_name, profile in profiles.items():
        path = checkpoint_dir / f"candidate_{profile_name}_seed42.pt"
        candidate_paths[profile_name] = path
        diagnostics = make_diagnostics(
            diagnostics_path,
            run_id=run_id,
            profile=profile_name,
            seed=42,
            effective_config=_model_config(config, profile),
            extreme_weight=float(config["training"].get("extreme_weight", 1.0)),
            extreme_threshold=float(config["training"].get("extreme_threshold", -1.0)),
        )
        model, selection, history = _fit_profile(
            config,
            profile_name,
            profile,
            train_data,
            val_data,
            sites,
            device,
            42,
            path,
            diagnostics,
        )
        selection["config"] = _model_config(config, profile)
        candidate_selection.append(selection)
        history_records.extend(history)
        qpred = _validation_outputs(model, val_data, device)
        metric_frame, conf, classes = _evaluate_predictions(
            val_data,
            qpred,
            split="validation",
            seed=42,
            run_tag=f"candidate_{profile_name}",
            sites=sites,
        )
        validation_records.append(metric_frame)
        candidate_confusions.update(conf)
        candidate_class_counts.extend(classes)

    candidate_selection.sort(
        key=lambda row: (
            -row["best_validation_primary_site_mean_mse_skill"],
            row["selected_epoch"],
            row["profile"],
        )
    )
    selected = candidate_selection[0]
    selected_profile = selected["profile"]
    selected_parameters = selected["profile_parameters"]
    selected_config = _model_config(config, selected_parameters)
    shutil.copy2(candidate_paths[selected_profile], checkpoint_dir / "selected_seed_42.pt")
    selected_checkpoint_paths = {42: checkpoint_dir / "selected_seed_42.pt"}
    final_validation_frames = []
    seed42_model = MultiScaleForecaster(selected_config).to(device)
    seed42_model.load_state_dict(torch.load(selected_checkpoint_paths[42], map_location=device, weights_only=True))
    seed42_qpred = _validation_outputs(seed42_model, val_data, device)
    seed42_metrics, selected_confusions, selected_class_counts = _evaluate_predictions(
        val_data,
        seed42_qpred,
        split="validation",
        seed=42,
        run_tag="selected_multiseed",
        sites=sites,
    )
    final_validation_frames.append(seed42_metrics)
    final_validation_frames.extend(validation_records)
    training_history = list(history_records)

    # Continue the frozen selected profile with four more predeclared seeds.
    for seed in (value for value in seeds if value != 42):
        path = checkpoint_dir / f"selected_seed_{seed}.pt"
        diagnostics = make_diagnostics(
            diagnostics_path,
            run_id=run_id,
            profile=selected_profile,
            seed=int(seed),
            effective_config=selected_config,
            extreme_weight=float(config["training"].get("extreme_weight", 1.0)),
            extreme_threshold=float(config["training"].get("extreme_threshold", -1.0)),
        )
        model, selection, history = _fit_profile(
            config,
            selected_profile,
            selected_parameters,
            train_data,
            val_data,
            sites,
            device,
            int(seed),
            path,
            diagnostics,
        )
        selection["config"] = selected_config
        selection["profile"] = selected_profile
        selected.setdefault("multiseed_runs", []).append(selection)
        training_history.extend(history)
        qpred = _validation_outputs(model, val_data, device)
        metric_frame, conf, classes = _evaluate_predictions(
            val_data,
            qpred,
            split="validation",
            seed=int(seed),
            run_tag="selected_multiseed",
            sites=sites,
        )
        final_validation_frames.append(metric_frame)
        selected_confusions.update(conf)
        selected_class_counts.extend(classes)
        selected_checkpoint_paths[int(seed)] = path

    # Freeze every choice and code/data/checkpoint hash before the only test pass.
    checkpoint_info = {
        str(seed): {
            "path": str(path),
            "sha256": sha256(path),
            "validation_selected_epoch": (
                selected["selected_epoch"]
                if seed == 42
                else next(
                    row["selected_epoch"]
                    for row in selected["multiseed_runs"]
                    if int(row["seed"]) == int(seed)
                )
            ),
        }
        for seed, path in selected_checkpoint_paths.items()
    }
    metadata = {
        "experiment": "DroughtCast Phase 2.2 proposed neural model",
        "run_id": run_id,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "dataset_protocol_version": DATASET_PROTOCOL_VERSION,
        "protocol_version": DATASET_PROTOCOL_VERSION,
        "normalizer_path": str(NORMALIZER_PATH),
        "normalizer_sha256": normalizer_hash,
        "normalizer_preprocessing_fingerprint": normalizer_metadata["preprocessing_fingerprint"],
        "configuration_path": str(config_path),
        "configuration_sha256": hashlib.sha256(config_bytes_before).hexdigest(),
        "feature_order": list(MODEL_FEATURES),
        "sites": sites,
        "split_boundaries": SPLIT_BOUNDARIES,
        "window_counts": window_counts,
        "window_date_ranges": baseline_meta["split_window_audit"]["date_ranges"],
        "seq_len": 6,
        "forecast_steps": 4,
        "target_scales": list(TARGETS),
        "output_per_scale": "[batch, forecast_lead, quantile]",
        "quantiles": list(QUANTILES),
        "model_summary": {
            "encoder": f"shared LSTM({selected_parameters['hidden_dim_1']}) -> dropout({selected_parameters['dropout']}) -> LSTM({selected_parameters['hidden_dim_2']})",
            "attention": "one shared additive temporal attention over six input steps",
            "heads": "three independent direct scale-specific quantile heads",
            "cascaded_decoder": False,
            "variable_gate": False,
        },
        "candidate_profiles": profiles,
        "candidate_selection": candidate_selection,
        "selected_profile": selected_profile,
        "selected_parameters": selected_parameters,
        "selected_config": selected_config,
        "selection_metric": "equal-site mean SPEI-3 MSE skill versus persistence over T+3 and T+4; validation-only",
        "training_loss": "sum of three target-scale quantile pinball losses; extreme_weight and extreme_threshold from canonical training config",
        "early_stopping": "validation-only primary skill; patience and min_delta recorded in selected config and source",
        "seeds": list(seeds),
        "selected_checkpoints": checkpoint_info,
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
            "packages": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "numpy": importlib.metadata.version("numpy"),
                "pandas": importlib.metadata.version("pandas"),
                "scikit-learn": importlib.metadata.version("scikit-learn"),
            },
        },
        "run_script_sha256_at_test_freeze": script_hash,
        "evaluation_code_sha256_at_test_freeze": script_hash,
        "baseline_run_id": baseline_meta["run_id"],
        "baseline_metadata_sha256": sha256(baseline_run / "baseline_experiment_metadata.json"),
        "baseline_metrics_sha256": sha256(baseline_run / "baseline_metrics.csv"),
        "test_set_was_consumed_in_phase2_1": True,
        "test_used_for_neural_training_selection": False,
        "test_evaluation_frozen": True,
        "test_evaluation_performed": False,
        "test_evaluation_note": "One locked final evaluation across the five frozen seed checkpoints after this manifest was written.",
        "verification": verification,
        "model_training": True,
    }
    metadata_path = result_dir / "neural_experiment_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    hyperparameters = {
        "candidate_profiles": profiles,
        "selected_profile": selected_profile,
        "selected_parameters": selected_parameters,
        "selected_epoch_seed42": selected["selected_epoch"],
        "candidate_selection": candidate_selection,
        "seeds": list(seeds),
        "selection_criterion": metadata["selection_metric"],
        "early_stopping_patience": int(config["training"].get("early_stopping_patience", 12)),
        "loss": metadata["training_loss"],
        "scheduler": "ReduceLROnPlateau(mode=max) on validation primary skill; factor=0.5",
        "gradient_clip_norm": float(config["training"].get("grad_clip", 1.0)),
    }
    (result_dir / "neural_hyperparameters.json").write_text(json.dumps(hyperparameters, indent=2), encoding="utf-8")
    pd.DataFrame(training_history).to_csv(result_dir / "neural_training_history.csv", index=False)
    validation_frame = pd.concat(final_validation_frames, ignore_index=True)
    validation_frame.to_csv(result_dir / "neural_metrics_validation.csv", index=False)
    pd.DataFrame(
        candidate_confusions | selected_confusions
    ).to_json(result_dir / "neural_confusion_matrices.json", orient="records", indent=2)
    pd.DataFrame(candidate_class_counts + selected_class_counts).to_csv(
        result_dir / "neural_class_distribution_validation.csv", index=False
    )
    all_class_counts = _class_distribution(train_data, "train", sites) + _class_distribution(val_data, "validation", sites)
    (result_dir / "neural_model_summary.txt").write_text(
        "DroughtCast protocol-10 quantile model\n"
        f"Run: {run_id}\n"
        f"Selected profile: {selected_profile}\n"
        "Input: [batch, 6, 15]\n"
        "Encoder: shared LSTM -> dropout -> LSTM -> shared additive temporal attention\n"
        "Scale heads: SPEI-3, SPEI-6, SPEI-12 direct quantile decoders\n"
        "Each head output: [batch, 4 leads, 3 quantiles (q10,q50,q90)]\n"
        "Median stacked point output: [batch, 3 scales, 4 leads]\n"
        f"Trainable parameters: {sum(parameter.numel() for parameter in seed42_model.parameters() if parameter.requires_grad)}\n",
        encoding="utf-8",
    )
    (result_dir / "neural_leakage_audit.json").write_text(
        json.dumps(
            {
                "baseline_window_audit": baseline_leakage,
                "window_counts": window_counts,
                "training_gradient_split": "train only",
                "profile_and_checkpoint_selection_split": "validation only",
                "test_target_values_accessed_before_freeze": False,
                "test_values_used_for_gradient_or_selection": False,
                "test_sample_count_alignment_checked_against_baselines": True,
                "normalizer_training_split_only": normalizer_metadata["training_split"] == "train",
                "normalizer_sha256": normalizer_hash,
                "same_month_spei_observation_features_match_targets": baseline_leakage["target_observation_feature_audit"],
                "future_climate_index_dates": {
                    name: {
                        "future_source_month_count": value["future_source_month_count"],
                        "future_effective_month_count": value["future_effective_month_count"],
                        "all_checks_pass": value["all_checks_pass"],
                    }
                    for name, value in baseline_leakage["index_provenance_audit"].items()
                },
                "missing_input_windows": baseline_leakage["input_missingness_by_split"],
                "test_set_consumed_in_phase2_1": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (result_dir / "neural_class_distribution.csv").write_text(
        pd.DataFrame(all_class_counts).to_csv(index=False), encoding="utf-8"
    )

    # The explicit freeze file is written only after every seed checkpoint,
    # validation metric, selected epoch and code/data hash is fixed.
    freeze = {
        "frozen_before_test_prediction": True,
        "run_id": run_id,
        "selected_profile": selected_profile,
        "selected_config": selected_config,
        "selected_checkpoints": checkpoint_info,
        "dataset_sha256": dataset_hash,
        "normalizer_sha256": normalizer_hash,
        "baseline_metrics_sha256": metadata["baseline_metrics_sha256"],
        "evaluation_code_sha256": script_hash,
        "test_split": {"start": SPLIT_BOUNDARIES["test_start"], "end": SPLIT_BOUNDARIES["test_end"]},
        "test_predictions_repeated": False,
        "post_test_model_changes_allowed": False,
    }
    (result_dir / "neural_test_evaluation_freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    # The test split is first materialized after every selection decision and
    # checkpoint hash is frozen.
    test_data = load_site_table(
        dataset_path,
        features=list(MODEL_FEATURES),
        seq_len=6,
        forecast_steps=4,
        metadata_path=NORMALIZER_PATH,
        site_filter=list(data_cfg["site_filter"]),
        site_specs=config.get("sites"),
        split_config=SPLIT_BOUNDARIES,
    )["test"]
    if len(test_data) != 1188:
        raise RuntimeError(
            f"Locked test window count differs from the benchmark: {len(test_data)}"
        )
    _verify_baseline_sample_counts(baseline_run, test_data, sites)

    # Final locked test pass: load each saved, validation-selected checkpoint;
    # do not fit, tune, threshold, or adjust anything from these predictions.
    test_metrics_frames, test_confusions, test_class_counts = [], {}, []
    for seed, checkpoint_path in selected_checkpoint_paths.items():
        model = MultiScaleForecaster(selected_config).to(device)
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        test_predictions = _quantile_predictions(model, test_data.X, device)
        metric_frame, conf, classes = _evaluate_predictions(
            test_data,
            test_predictions,
            split="test",
            seed=int(seed),
            run_tag="locked_selected_profile",
            sites=sites,
        )
        test_metrics_frames.append(metric_frame)
        test_confusions.update(conf)
        test_class_counts.extend(classes)
    test_frame = pd.concat(test_metrics_frames, ignore_index=True)
    test_frame.to_csv(result_dir / "neural_metrics_test.csv", index=False)
    test_frame.to_csv(result_dir / "neural_per_site_metrics.csv", index=False)
    (result_dir / "neural_confusion_matrices.json").write_text(
        json.dumps(candidate_confusions | selected_confusions | test_confusions, indent=2), encoding="utf-8"
    )
    pd.DataFrame(candidate_class_counts + selected_class_counts + test_class_counts).to_csv(
        result_dir / "neural_class_distribution.csv", index=False
    )
    comparison = _make_baseline_comparison(result_dir, baseline_run, test_frame)
    metadata["test_evaluation_performed"] = True
    metadata["test_evaluation_checkpoint_hashes"] = {
        str(seed): sha256(path) for seed, path in selected_checkpoint_paths.items()
    }
    metadata["test_metrics_path"] = str(result_dir / "neural_metrics_test.csv")
    metadata["test_predictions_repeated"] = False
    (result_dir / "neural_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    _report(result_dir, metadata, validation_frame, test_frame, comparison)

    # Confirm all immutable protocol inputs still have the original hashes.
    if sha256(dataset_path) != dataset_hash or sha256(NORMALIZER_PATH) != normalizer_hash:
        raise RuntimeError("Canonical data or verified normalizer changed during Phase 2.2")
    if config_bytes_before != Path(config_path).read_bytes():
        raise RuntimeError("Canonical config changed during Phase 2.2")
    print(json.dumps({"run_id": run_id, "result_dir": str(result_dir), "model_dir": str(checkpoint_dir), "selected_profile": selected_profile, "selected_epoch_seed42": selected["selected_epoch"], "window_counts": window_counts, "test_metrics_rows": len(test_frame), "comparison_rows": len(comparison)}, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--result-root", type=Path, default=Path("results/maharashtra/phase2/neural"))
    parser.add_argument("--model-root", type=Path, default=Path("models/neural"))
    parser.add_argument("--baseline-run", type=Path, default=BASELINE_RUN)
    parser.add_argument(
        "--diagnostics-dir", type=Path, default=None,
        help="Optional new directory for detached per-epoch training/validation diagnostics; disabled by default.",
    )
    args = parser.parse_args()
    run(
        args.config, args.result_root, args.model_root, args.baseline_run,
        diagnostics_dir=args.diagnostics_dir,
    )


if __name__ == "__main__":
    main()
