"""Run classical Phase 2.1 baselines without invoking DroughtCast's model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.loader import load_site_table, validate_dataset_protocol
from data.preprocessing import (
    validate_normalizer_metadata,
)
from data.protocol import (
    DATASET_PROTOCOL_VERSION,
    INDEX_ALIGNMENT,
    MODEL_FEATURES,
    SPLIT_BOUNDARIES,
)

TARGETS = ("spei_3", "spei_6", "spei_12")
TARGET_LABELS = {
    "spei_3": "SPEI-3",
    "spei_6": "SPEI-6",
    "spei_12": "SPEI-12",
}
CLIMATE_INDICES = ("oni", "dmi", "nao", "pdo")
CATEGORY_NAMES = (
    "extreme_drought",
    "severe_drought",
    "moderate_drought",
    "near_normal",
    "moderately_wet",
    "very_wet",
    "extremely_wet",
)
MODEL_METHODS = ("ridge", "random_forest", "xgboost")
FEATURE_VARIANTS = ("all_features", "without_climate_indices")
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def drought_categories(values: np.ndarray) -> np.ndarray:
    """Use the project's seven fixed SPEI categories; -1 represents missing."""
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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    n = len(y_true)
    if n == 0:
        return {"n_valid": 0, "mae": None, "rmse": None, "r2": None, "pearson_r": None}
    corr = None
    if n >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(pearsonr(y_true, y_pred).statistic)
    return {
        "n_valid": n,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if n >= 2 and np.std(y_true) > 0 else None,
        "pearson_r": corr,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict, list[list[int]]]:
    y_class = drought_categories(y_true)
    p_class = drought_categories(y_pred)
    valid = (y_class >= 0) & (p_class >= 0)
    y_class, p_class = y_class[valid], p_class[valid]
    matrix = confusion_matrix(y_class, p_class, labels=np.arange(7))
    y_values = np.asarray(y_true, dtype=float)[valid]
    p_values = np.asarray(y_pred, dtype=float)[valid]
    y_severe = y_class == 1  # -2 < SPEI <= -1.5, the documented severe category.
    p_severe = p_class == 1
    y_severe_or_extreme = y_values <= -1.5
    p_severe_or_extreme = p_values <= -1.5
    return (
        {
            "category_accuracy": float(accuracy_score(y_class, p_class)) if len(y_class) else None,
            "macro_f1": float(f1_score(y_class, p_class, labels=np.arange(7), average="macro", zero_division=0)) if len(y_class) else None,
            "weighted_f1": float(f1_score(y_class, p_class, labels=np.arange(7), average="weighted", zero_division=0)) if len(y_class) else None,
            "severe_drought_precision": float(precision_score(y_severe, p_severe, zero_division=0)) if len(y_class) else None,
            "severe_drought_recall": float(recall_score(y_severe, p_severe, zero_division=0)) if len(y_class) else None,
            "severe_or_extreme_event_precision_le_minus_1_5": float(precision_score(y_severe_or_extreme, p_severe_or_extreme, zero_division=0)) if len(y_class) else None,
            "severe_or_extreme_event_recall_le_minus_1_5": float(recall_score(y_severe_or_extreme, p_severe_or_extreme, zero_division=0)) if len(y_class) else None,
        },
        matrix.astype(int).tolist(),
    )


def persistence_predictions(dataset, forecast_steps: int = 4) -> np.ndarray:
    """Repeat the latest finite observed SPEI within each input window."""
    result = np.full((len(dataset), 3, forecast_steps), np.nan, dtype=float)
    for row, metadata in enumerate(dataset.metadata):
        history = np.asarray(metadata.get("spei_history", []), dtype=float)
        if history.ndim != 2 or history.shape[1] != 3:
            continue
        for scale in range(3):
            valid = np.flatnonzero(np.isfinite(history[:, scale]))
            if len(valid):
                result[row, scale, :] = history[valid[-1], scale]
    return result


def climatology_predictions(train_dataset, target_dataset, raw_frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Site/calendar-month target means fit from unique training rows only."""
    train_end = pd.Timestamp(SPLIT_BOUNDARIES["train_end"])
    training = raw_frame.loc[raw_frame["date"] <= train_end].copy()
    for target in TARGETS:
        training[target] = pd.to_numeric(training[target], errors="coerce")
    means: dict[tuple[str, int, int], float] = {}
    site_means: dict[tuple[str, int], float] = {}
    global_means: dict[int, float] = {}
    for scale, target in enumerate(TARGETS):
        valid = training.loc[np.isfinite(training[target])]
        global_means[scale] = float(valid[target].mean())
        site_means.update(
            {
                (str(site), scale): float(values.mean())
                for site, values in valid.groupby("site")[target]
            }
        )
        month_groups = valid.assign(_month=valid["date"].dt.month).groupby(["site", "_month"])[target].mean()
        for (site, month), value in month_groups.items():
            means[(str(site), int(month), scale)] = float(value)

    predictions = np.full_like(target_dataset.Y.numpy(), np.nan, dtype=float)
    fallback_counts: Counter[str] = Counter()
    for row, metadata in enumerate(target_dataset.metadata):
        site = str(metadata["site"])
        for lead, target_date in enumerate(metadata["target_dates"]):
            month = int(target_date[5:7])
            for scale in range(3):
                key = (site, month, scale)
                if key in means:
                    predictions[row, scale, lead] = means[key]
                elif (site, scale) in site_means:
                    predictions[row, scale, lead] = site_means[(site, scale)]
                    fallback_counts["site_overall"] += 1
                else:
                    predictions[row, scale, lead] = global_means[scale]
                    fallback_counts["pooled_overall"] += 1
    return predictions, {
        "fit_period_end": str(train_end.date()),
        "unique_training_rows": len(training),
        "fallback_counts": dict(fallback_counts),
        "fallback_order": "site/calendar-month -> site overall -> pooled training overall",
    }


def _xgb_class():
    try:
        from xgboost import XGBRegressor
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return None, f"XGBoost import failed: {type(exc).__name__}: {exc}"
    return XGBRegressor, None


def candidate_configs(method: str) -> list[dict]:
    if method == "ridge":
        return [{"alpha": alpha} for alpha in (0.1, 1.0, 10.0, 100.0)]
    if method == "random_forest":
        return [
            {"n_estimators": 160, "max_depth": depth, "min_samples_leaf": leaf}
            for depth in (None, 12)
            for leaf in (2, 5)
        ]
    if method == "xgboost":
        return [
            {"n_estimators": trees, "max_depth": depth, "learning_rate": 0.05}
            for trees in (100, 200)
            for depth in (2, 4)
        ]
    raise ValueError(f"Unknown model method {method}")


def make_estimator(method: str, params: dict, seed: int):
    if method == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(params["alpha"])))
    if method == "random_forest":
        return RandomForestRegressor(
            **params,
            random_state=int(seed),
            n_jobs=-1,
        )
    if method == "xgboost":
        cls, error = _xgb_class()
        if cls is None:
            raise RuntimeError(error)
        return cls(
            **params,
            objective="reg:squarederror",
            random_state=int(seed),
            n_jobs=1,
            subsample=0.9,
            colsample_bytree=0.9,
            verbosity=0,
        )
    raise ValueError(f"Unknown model method {method}")


def select_estimator(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[object, dict]:
    """Select hyperparameters by validation RMSE; never accepts test arrays."""
    candidates = []
    for params in candidate_configs(method):
        estimator = make_estimator(method, params, seed)
        estimator.fit(x_train, y_train)
        prediction = estimator.predict(x_val)
        score = regression_metrics(y_val, prediction)["rmse"]
        candidates.append({"params": params, "validation_rmse": score, "estimator": estimator})
    candidates.sort(key=lambda item: (float("inf") if item["validation_rmse"] is None else item["validation_rmse"], json.dumps(item["params"], sort_keys=True)))
    selected = candidates[0]
    return selected["estimator"], {
        "selected_params": selected["params"],
        "selected_validation_rmse": selected["validation_rmse"],
        "candidate_validation_scores": [
            {"params": item["params"], "validation_rmse": item["validation_rmse"]}
            for item in candidates
        ],
        "fit_rows": len(x_train),
        "selection_rows": len(x_val),
        "selection_metric": "RMSE",
        "selection_split": "validation",
    }


def _validate_windows(datasets: dict, sites: list[str]) -> dict:
    signatures = {}
    per_site = {}
    max_train_target = None
    min_val_target = None
    min_test_target = None
    future_index_sources = 0
    source_records = 0
    future_index_effective = 0
    split_crossings = 0
    chronology_errors = 0
    target_feature_overlap = 0
    ranges = {}
    for split, dataset in datasets.items():
        current = set()
        counts = Counter()
        input_starts, input_ends, target_starts, target_ends = [], [], [], []
        for metadata in dataset.metadata:
            site = str(metadata["site"])
            counts[site] += 1
            dates = pd.to_datetime(metadata["dates"])
            targets = pd.to_datetime(metadata["target_dates"])
            if len(dates) != 6 or len(targets) != 4 or dates.is_monotonic_increasing is False:
                chronology_errors += 1
            if dates[-1] >= targets[0] or not targets.is_monotonic_increasing:
                target_feature_overlap += 1
            if not dates.is_unique or not targets.is_unique:
                chronology_errors += 1
            if not all(
                dates[index] + pd.offsets.MonthBegin(1) == dates[index + 1]
                for index in range(len(dates) - 1)
            ) or not all(
                targets[index] + pd.offsets.MonthBegin(1) == targets[index + 1]
                for index in range(len(targets) - 1)
            ):
                chronology_errors += 1
            if any(not _date_in_split(date, split) for date in [*dates, *targets]):
                split_crossings += 1
            all_dates = list(dates) + list(targets)
            if any(pd.Timestamp(d).strftime("%Y-%m-%d") != d.strftime("%Y-%m-%d") for d in all_dates):
                chronology_errors += 1
            signature = (site, tuple(d.strftime("%Y-%m-%d") for d in dates), tuple(d.strftime("%Y-%m-%d") for d in targets))
            if signature in current:
                chronology_errors += 1
            current.add(signature)
            input_starts.append(dates[0])
            input_ends.append(dates[-1])
            target_starts.append(targets[0])
            target_ends.append(targets[-1])
            if split == "train":
                max_train_target = max(max_train_target or targets[-1], targets[-1])
            elif split == "val":
                min_val_target = min(min_val_target or targets[0], targets[0])
            elif split == "test":
                min_test_target = min(min_test_target or targets[0], targets[0])
            provenance = metadata.get("index_provenance_by_index", [])
            for input_month, indices in zip(dates, provenance, strict=False):
                for name in CLIMATE_INDICES:
                    record = indices.get(name, {})
                    if record.get("is_missing"):
                        continue
                    source_records += 1
                    source_month = record.get("source_month")
                    effective_month = record.get("effective_month")
                    if source_month and pd.Timestamp(source_month) > input_month:
                        future_index_sources += 1
                    if effective_month and pd.Timestamp(effective_month) > input_month:
                        future_index_effective += 1
                    age = record.get("observation_age_months")
                    if age is not None and (
                        float(age) < 0
                        or float(age) > INDEX_ALIGNMENT[name]["max_age_months"]
                    ):
                        chronology_errors += 1
        signatures[split] = current
        per_site[split] = {site: int(counts.get(site, 0)) for site in sites}
        ranges[split] = {
            "input_start": min(input_starts).strftime("%Y-%m") if input_starts else None,
            "input_end": max(input_ends).strftime("%Y-%m") if input_ends else None,
            "target_start": min(target_starts).strftime("%Y-%m") if target_starts else None,
            "target_end": max(target_ends).strftime("%Y-%m") if target_ends else None,
        }
    cross_split_duplicates = sum(
        len(signatures[left].intersection(signatures[right]))
        for i, left in enumerate(SPLITS)
        for right in SPLITS[i + 1 :]
    )
    return {
        "window_counts_per_site": per_site,
        "window_counts": {split: len(datasets[split]) for split in SPLITS},
        "date_ranges": ranges,
        "input_months": 6,
        "target_months": 4,
        "max_train_target_month": None if max_train_target is None else max_train_target.strftime("%Y-%m"),
        "min_validation_target_month": None if min_val_target is None else min_val_target.strftime("%Y-%m"),
        "min_test_target_month": None if min_test_target is None else min_test_target.strftime("%Y-%m"),
        "split_crossings": split_crossings,
        "duplicate_windows_within_or_across_splits": chronology_errors + target_feature_overlap + cross_split_duplicates,
        "future_source_month_records": future_index_sources,
        "future_effective_month_records": future_index_effective,
        "audited_available_index_records": source_records,
        "no_future_targets_in_input_windows": target_feature_overlap == 0,
        "no_cross_split_window_detected": split_crossings == 0 and chronology_errors == 0 and cross_split_duplicates == 0,
        "no_future_climate_index_dates": future_index_sources == 0 and future_index_effective == 0,
    }


def _metric_row(method, variant, target, lead, site, y, p, target_mask) -> dict:
    valid = np.asarray(target_mask, dtype=bool) & np.isfinite(y) & np.isfinite(p)
    regression = regression_metrics(np.asarray(y)[valid], np.asarray(p)[valid])
    classification, matrix = classification_metrics(np.asarray(y)[valid], np.asarray(p)[valid])
    row = {
        "model": method,
        "feature_variant": variant,
        "target": TARGET_LABELS[target],
        "target_column": target,
        "lead": f"T+{lead + 1}",
        "site": site,
        "n_total_windows": len(y),
        "n_valid": int(regression["n_valid"]),
        "n_excluded_missing_target_or_prediction": int(len(y) - regression["n_valid"]),
        **regression,
        **classification,
    }
    return row, matrix


def _metric_tables(predictions, datasets, sites):
    rows = []
    confusion = {}
    class_dist = []
    for split in SPLITS:
        ds = datasets[split]
        y_all = ds.Y.numpy()
        mask_all = ds.mask.numpy()
        metadata = ds.metadata
        for scale_index, target in enumerate(TARGETS):
            for lead in range(y_all.shape[2]):
                y = y_all[:, scale_index, lead]
                valid_target = mask_all[:, scale_index, lead]
                split_sites = np.asarray([str(m["site"]) for m in metadata])
                for site in [*sites, "ALL"]:
                    site_mask = np.ones(len(metadata), dtype=bool) if site == "ALL" else split_sites == site
                    selected_cats = drought_categories(y[site_mask & valid_target])
                    counts = np.bincount(selected_cats[selected_cats >= 0], minlength=7)
                    for cat, name in enumerate(CATEGORY_NAMES):
                        class_dist.append(
                            {
                                "split": split,
                                "site": site,
                                "target": TARGET_LABELS[target],
                                "target_column": target,
                                "lead": f"T+{lead + 1}",
                                "category_code": cat,
                                "category": name,
                                "count": int(counts[cat]),
                                "valid_target_count": len(selected_cats),
                                "n_windows": int(site_mask.sum()),
                                "missing_target_count": int((site_mask & ~valid_target).sum()),
                            }
                        )
        if split != "test":
            continue
        for key, pred in predictions.items():
            method, variant = key
            pred = np.asarray(pred, dtype=float)
            for scale_index, target in enumerate(TARGETS):
                for lead in range(y_all.shape[2]):
                    y = y_all[:, scale_index, lead]
                    p = pred[:, scale_index, lead]
                    valid_target = mask_all[:, scale_index, lead]
                    split_sites = np.asarray([str(m["site"]) for m in metadata])
                    for site in [*sites, "ALL"]:
                        site_mask = np.ones(len(metadata), dtype=bool) if site == "ALL" else split_sites == site
                        row, matrix = _metric_row(
                            method,
                            variant,
                            target,
                            lead,
                            site,
                            y[site_mask],
                            p[site_mask],
                            valid_target[site_mask],
                        )
                        rows.append(row)
                        confusion[f"{method}|{variant}|{target}|T+{lead + 1}|{site}"] = {
                            "labels": list(CATEGORY_NAMES),
                            "counts": matrix,
                            "class_order": "true rows, predicted columns",
                            "severe_event_threshold": "SPEI <= -1.5",
                        }
    return pd.DataFrame(rows), pd.DataFrame(class_dist), confusion


def _environment() -> dict:
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "joblib", "torch"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages}


def _date_in_split(date: pd.Timestamp, split: str) -> bool:
    if split == "train":
        return date <= pd.Timestamp(SPLIT_BOUNDARIES["train_end"])
    if split == "val":
        return pd.Timestamp(SPLIT_BOUNDARIES["val_start"]) <= date <= pd.Timestamp(SPLIT_BOUNDARIES["val_end"])
    return pd.Timestamp(SPLIT_BOUNDARIES["test_start"]) <= date <= pd.Timestamp(SPLIT_BOUNDARIES["test_end"])


def run(config_path: Path, output_root: Path, model_root: Path, seed: int = 42) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    features = list(data_cfg["features"])
    if features != MODEL_FEATURES:
        raise ValueError("Baseline config feature order does not match protocol 10")
    if int(data_cfg["seq_len"]) != 6 or int(data_cfg["forecast_steps"]) != 4:
        raise ValueError("Phase 2.1 implementation expects the configured 6-by-4 window contract")
    dataset_path = Path(data_cfg["path"])
    normalizer_path = Path("data/normalizers/maharashtra_protocol10.json")
    normalizer_hash = sha256(normalizer_path)
    dataset_hash = sha256(dataset_path)
    normalizer_metadata = json.loads(normalizer_path.read_text(encoding="utf-8"))
    validate_normalizer_metadata(
        normalizer_metadata,
        features,
        expected_dataset_sha256=dataset_hash,
        expected_protocol_version=DATASET_PROTOCOL_VERSION,
        expected_split_boundaries=SPLIT_BOUNDARIES,
        expected_seq_len=6,
        expected_forecast_steps=4,
    )
    validate_dataset_protocol(
        dataset_path.parent / "dataset_metadata.json",
        data_cfg.get("target_source", "local"),
        features,
    )
    # Passing the verified artifact makes load_site_table reject any changed fit
    # statistics; its independently fitted train-only values must match exactly.
    datasets = load_site_table(
        dataset_path,
        features,
        6,
        4,
        normalizer_path,
        site_filter=data_cfg["site_filter"],
        site_specs=config.get("sites"),
        split_config=data_cfg.get("split_boundaries"),
    )
    if sha256(normalizer_path) != normalizer_hash:
        raise RuntimeError("Verified normalizer artifact changed while loading windows")
    raw_frame = pd.read_csv(dataset_path, parse_dates=["date"])
    raw_frame["site"] = raw_frame["site"].astype(str).str.strip().str.lower()
    sites = sorted(data_cfg["site_filter"])
    windows_audit = _validate_windows(datasets, sites)
    if not all(
        windows_audit[key]
        for key in (
            "no_future_targets_in_input_windows",
            "no_cross_split_window_detected",
            "no_future_climate_index_dates",
        )
    ):
        raise RuntimeError(f"Baseline window leakage audit failed: {windows_audit}")
    for split in SPLITS:
        if not np.isfinite(datasets[split].X.numpy()).all():
            raise RuntimeError(f"Non-finite normalized features in {split} inputs")

    output_root.mkdir(parents=True, exist_ok=True)
    script_hash = sha256(Path(__file__))
    run_id = f"protocol{DATASET_PROTOCOL_VERSION}_seed{seed}_{dataset_hash[:10]}_{normalizer_hash[:10]}_{script_hash[:8]}"
    run_models_dir = model_root / run_id
    run_output_dir = output_root / run_id
    if run_models_dir.exists() or run_output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite baseline run {run_id}")
    run_models_dir.mkdir(parents=True)
    run_output_dir.mkdir(parents=True)

    xgb_cls, xgb_error = _xgb_class()
    methods = ["persistence", "seasonal_climatology", *MODEL_METHODS]
    if xgb_cls is None:
        methods.remove("xgboost")
    feature_indices = {
        "all_features": np.arange(len(features)),
        "without_climate_indices": np.asarray(
            [i for i, name in enumerate(features) if name not in CLIMATE_INDICES]
        ),
    }
    hyperparameters = {
        "seed": int(seed),
        "selection_metric": "validation RMSE for each target-scale/lead/feature-variant model",
        "models": {},
        "xgboost_status": "NOT RUN" if xgb_cls is None else "RUN",
        "xgboost_reason": xgb_error,
        "candidate_grids": {method: candidate_configs(method) for method in MODEL_METHODS if method != "xgboost" or xgb_cls is not None},
        "climatology": None,
    }
    validation_preds = {}
    selected_models = {}

    # Train and tune exclusively with train/validation. Test windows are not
    # indexed until all candidate selection and artifact saves are complete.
    for method in methods:
        if method == "persistence":
            for variant in FEATURE_VARIANTS:
                validation_preds[(method, variant)] = persistence_predictions(datasets["val"])
            continue
        if method == "seasonal_climatology":
            val_pred, clim_info = climatology_predictions(datasets["train"], datasets["val"], raw_frame)
            hyperparameters["climatology"] = clim_info
            for variant in FEATURE_VARIANTS:
                validation_preds[(method, variant)] = val_pred.copy()
            continue
        for variant in FEATURE_VARIANTS:
            indices = feature_indices[variant]
            x_train_all = datasets["train"].X.numpy()[:, :, indices].reshape(len(datasets["train"]), -1)
            x_val_all = datasets["val"].X.numpy()[:, :, indices].reshape(len(datasets["val"]), -1)
            y_train_all = datasets["train"].Y.numpy()
            mask_train_all = datasets["train"].mask.numpy()
            y_val_all = datasets["val"].Y.numpy()
            mask_val_all = datasets["val"].mask.numpy()
            val_pred = np.full_like(y_val_all, np.nan, dtype=float)
            for scale_index, target in enumerate(TARGETS):
                for lead in range(4):
                    train_mask = mask_train_all[:, scale_index, lead]
                    val_mask = mask_val_all[:, scale_index, lead]
                    if train_mask.sum() < 10 or not val_mask.any():
                        raise RuntimeError(f"Insufficient train/validation targets for {method} {target} T+{lead + 1}")
                    estimator, selection = select_estimator(
                        method,
                        x_train_all[train_mask],
                        y_train_all[train_mask, scale_index, lead],
                        x_val_all[val_mask],
                        y_val_all[val_mask, scale_index, lead],
                        seed,
                    )
                    key = (method, variant, scale_index, lead)
                    selected_models[key] = estimator
                    val_pred[:, scale_index, lead] = estimator.predict(x_val_all)
                    hyperparameters["models"]["|".join(map(str, key))] = {
                        **selection,
                        "fit_target_max_date": max(
                            m["target_dates"][lead]
                            for m, valid in zip(datasets["train"].metadata, train_mask, strict=True)
                            if valid
                        ),
                        "validation_targets_used": int(val_mask.sum()),
                    }
                    model_file = run_models_dir / f"{method}_{variant}_{target}_lead{lead + 1}.joblib"
                    joblib.dump(
                        {
                            "estimator": estimator,
                            "metadata": {
                                "method": method,
                                "feature_variant": variant,
                                "target": target,
                                "lead": lead + 1,
                                "selected_params": selection["selected_params"],
                                "seed": int(seed),
                                "dataset_sha256": dataset_hash,
                                "normalizer_sha256": normalizer_hash,
                            },
                        },
                        model_file,
                    )
            validation_preds[(method, variant)] = val_pred

    # Validation is used for selection only. Persist this marker before the
    # first test prediction to make the evaluation boundary auditable.
    selection_complete = {
        "selection_complete_before_test_evaluation": True,
        "selected_model_count": len(selected_models),
        "selected_keys": ["|".join(map(str, key)) for key in selected_models],
    }
    (run_output_dir / "selection_complete.json").write_text(
        json.dumps(selection_complete, indent=2), encoding="utf-8"
    )

    predictions = {}
    for method in methods:
        if method == "persistence":
            pred = persistence_predictions(datasets["test"])
            for variant in FEATURE_VARIANTS:
                predictions[(method, variant)] = pred.copy()
        elif method == "seasonal_climatology":
            pred, _ = climatology_predictions(datasets["train"], datasets["test"], raw_frame)
            for variant in FEATURE_VARIANTS:
                predictions[(method, variant)] = pred.copy()
        else:
            for variant in FEATURE_VARIANTS:
                indices = feature_indices[variant]
                x_test = datasets["test"].X.numpy()[:, :, indices].reshape(len(datasets["test"]), -1)
                pred = np.full_like(datasets["test"].Y.numpy(), np.nan, dtype=float)
                for scale_index, scale in enumerate(TARGETS):
                    for lead in range(4):
                        estimator = selected_models[(method, variant, scale_index, lead)]
                        pred[:, scale_index, lead] = estimator.predict(x_test)
                predictions[(method, variant)] = pred

    metrics, class_dist, confusion = _metric_tables(predictions, datasets, sites)
    metrics_path = run_output_dir / "baseline_metrics.csv"
    per_site_path = run_output_dir / "baseline_per_site_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    metrics.loc[metrics["site"] != "ALL"].to_csv(per_site_path, index=False)
    class_dist.to_csv(run_output_dir / "baseline_class_distribution.csv", index=False)
    (run_output_dir / "baseline_confusion_matrices.json").write_text(
        json.dumps(confusion, indent=2), encoding="utf-8"
    )
    (run_output_dir / "baseline_hyperparameters.json").write_text(
        json.dumps(hyperparameters, indent=2, allow_nan=False), encoding="utf-8"
    )

    sensitivity_rows = []
    for method in methods:
        for target in TARGETS:
            for lead in range(1, 5):
                all_row = metrics.loc[
                    (metrics["model"] == method)
                    & (metrics["feature_variant"] == "all_features")
                    & (metrics["target_column"] == target)
                    & (metrics["lead"] == f"T+{lead}")
                    & (metrics["site"] == "ALL")
                ]
                no_index_row = metrics.loc[
                    (metrics["model"] == method)
                    & (metrics["feature_variant"] == "without_climate_indices")
                    & (metrics["target_column"] == target)
                    & (metrics["lead"] == f"T+{lead}")
                    & (metrics["site"] == "ALL")
                ]
                if all_row.empty or no_index_row.empty:
                    continue
                left, right = all_row.iloc[0], no_index_row.iloc[0]
                sensitivity_rows.append(
                    {
                        "model": method,
                        "target": TARGET_LABELS[target],
                        "lead": f"T+{lead}",
                        "all_features_rmse": left["rmse"],
                        "without_climate_indices_rmse": right["rmse"],
                        "rmse_change_without_indices_minus_all": (
                            right["rmse"] - left["rmse"]
                            if pd.notna(right["rmse"]) and pd.notna(left["rmse"])
                            else None
                        ),
                        "all_features_category_accuracy": left["category_accuracy"],
                        "without_indices_category_accuracy": right["category_accuracy"],
                        "all_features_n": int(left["n_valid"]),
                        "without_indices_n": int(right["n_valid"]),
                        "ablation": "oni,dmi,nao,pdo removed; same windows and verified canonical normalizer; not a replacement protocol",
                    }
                )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(run_output_dir / "baseline_sensitivity_analysis.csv", index=False)

    metadata = {
        "experiment": "DroughtCast Phase 2.1 classical baselines",
        "run_id": run_id,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "normalizer_path": str(normalizer_path),
        "normalizer_sha256": normalizer_hash,
        "experiment_script_sha256": script_hash,
        "normalizer_preprocessing_fingerprint": normalizer_metadata["preprocessing_fingerprint"],
        "protocol_version": DATASET_PROTOCOL_VERSION,
        "features": features,
        "feature_order_matches_protocol": features == MODEL_FEATURES,
        "feature_variants": {
            "all_features": features,
            "without_climate_indices": [x for x in features if x not in CLIMATE_INDICES],
        },
        "target_scales": list(TARGETS),
        "seq_len": 6,
        "forecast_steps": 4,
        "split_boundaries": SPLIT_BOUNDARIES,
        "split_window_audit": windows_audit,
        "window_counts": {split: len(datasets[split]) for split in SPLITS},
        "missing_targets_by_split_scale_lead": {
            split: {
                target: [
                    int((~datasets[split].mask.numpy()[:, scale_index, lead]).sum())
                    for lead in range(4)
                ]
                for scale_index, target in enumerate(TARGETS)
            }
            for split in SPLITS
        },
        "target_alignment": "six contiguous monthly inputs; four immediate following monthly targets; all windows wholly within one protocol split",
        "missing_value_policy": "invalid sentinels converted to NaN; verified per-site train-only normalizer; non-finite inputs mean-imputed from training statistics; target masks exclude missing labels",
        "normalization_fit_period": "1990-01 through 2010-12, per site, train rows only",
        "seed": int(seed),
        "model_selection": "each Ridge/RF/XGBoost target-scale-lead-variant model selected by validation RMSE; fitted on train only; no test values used until all selections and model artifacts were saved",
        "test_evaluation_consumed": True,
        "test_used_for_fit_or_selection": False,
        "drought_categories": dict(enumerate(CATEGORY_NAMES)),
        "severe_category_definition": "-2.0 < SPEI <= -1.5; extreme drought (SPEI <= -2.0) is a separate category",
        "severe_or_extreme_event_definition": "SPEI <= -1.5",
        "methods": methods,
        "xgboost_status": "NOT RUN" if xgb_cls is None else "RUN",
        "xgboost_reason": xgb_error,
        "environment": _environment(),
        "model_forward_pass": False,
        "proposed_droughtcast_neural_model_trained": False,
    }
    (run_output_dir / "baseline_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    (run_output_dir / "baseline_leakage_audit.json").write_text(
        json.dumps(
            {
                **windows_audit,
                **selection_complete,
                "fit_splits": ["train"],
                "hyperparameter_selection_split": "validation",
                "evaluation_split": "test",
                "test_used_before_selection_complete": False,
                "normalizer_sha256": normalizer_hash,
                "normalizer_training_split_only": normalizer_metadata["training_split"] == "train",
                "normalizer_fit_period": normalizer_metadata["training_dates_by_site"],
                "legacy_normalizer_used": False,
                "target_derived_features_limited_to_input_dates": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    overall = metrics.loc[metrics["site"] == "ALL"]
    scale_summary = (
        overall.groupby(["model", "feature_variant", "target_column"], as_index=False)
        .agg(mean_lead_rmse=("rmse", "mean"), mean_lead_mae=("mae", "mean"), mean_lead_f1=("macro_f1", "mean"))
    )
    best_rows = []
    for target in TARGETS:
        selected = scale_summary.loc[
            (scale_summary["target_column"] == target)
            & (scale_summary["feature_variant"] == "all_features")
        ].sort_values("mean_lead_rmse")
        if not selected.empty:
            best_rows.append(selected.iloc[0].to_dict())
    best = pd.DataFrame(best_rows)
    model_scale_summary = scale_summary.loc[
        scale_summary["feature_variant"] == "all_features"
    ]
    persistence_comparisons = []
    for target in TARGETS:
        base = model_scale_summary.loc[
            (model_scale_summary["target_column"] == target)
            & (model_scale_summary["model"] == "persistence"),
            "mean_lead_rmse",
        ]
        if base.empty:
            continue
        for method in MODEL_METHODS:
            candidate = model_scale_summary.loc[
                (model_scale_summary["target_column"] == target)
                & (model_scale_summary["model"] == method),
                "mean_lead_rmse",
            ]
            if not candidate.empty:
                persistence_comparisons.append(
                    {
                        "target": target,
                        "model": method,
                        "mean_lead_rmse": float(candidate.iloc[0]),
                        "persistence_mean_lead_rmse": float(base.iloc[0]),
                        "delta_rmse_vs_persistence": float(candidate.iloc[0] - base.iloc[0]),
                    }
                )
    severe_summary = (
        overall.groupby(["model", "feature_variant", "target_column"], as_index=False)
        .agg(mean_lead_severe_precision=("severe_drought_precision", "mean"), mean_lead_severe_recall=("severe_drought_recall", "mean"), mean_lead_severe_or_extreme_recall=("severe_or_extreme_event_recall_le_minus_1_5", "mean"))
    )
    site_spread = (
        metrics.loc[metrics["site"] != "ALL"]
        .groupby(["model", "feature_variant", "target_column"], as_index=False)
        .agg(site_rmse_min=("rmse", "min"), site_rmse_max=("rmse", "max"), site_rmse_mean=("rmse", "mean"))
    )
    weak = site_spread.sort_values("site_rmse_max", ascending=False).head(8)
    lines = [
        "# Phase 2.1 — Classical Baseline Experiment Report",
        "",
        "**Status: BASELINES COMPLETE WITH LIMITATIONS**  ",
        f"Run: `{run_id}`. The proposed DroughtCast neural architecture was not trained and its forward pass was not executed.",
        "",
        "## Experiment specification",
        "",
        f"- Sites: {len(sites)} Maharashtra locations; input features: {len(features)} in protocol order.",
        "- Input/target contract: six input months, four leads (T+1–T+4), targets SPEI-3/6/12.",
        f"- Windows: train {len(datasets['train'])}, validation {len(datasets['val'])}, test {len(datasets['test'])}.",
        "- Split boundaries: train through 2010-12; validation 2012-01–2015-12; test 2017-01–2025-12; gaps retained.",
        f"- Verified normalizer SHA-256: `{normalizer_hash}`. Dataset SHA-256: `{dataset_hash}`.",
        "- Baselines: persistence, site/calendar-month training climatology, Ridge, Random Forest, and XGBoost when available.",
        "- Ridge/RF/XGBoost use pooled training windows, fit per target scale and lead, and select hyperparameters by validation RMSE. The selected estimators remain trained on training windows only.",
        "- Feature ablation removes ONI, DMI, NAO and PDO while keeping the same windows and original 15-feature training normalizer; it is a sensitivity experiment only.",
        "",
        "## Test results by target scale",
        "",
        "Lowest pooled test RMSE after averaging the four leads is descriptive only; method configurations were selected on validation, not on test.",
        "",
        "| Target | Lowest descriptive test RMSE | Mean lead RMSE | Mean lead MAE | Mean lead macro F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in best.to_dict("records"):
        lines.append(
            f"| {TARGET_LABELS[row['target_column']]} | {row['model']} | {row['mean_lead_rmse']:.4f} | {row['mean_lead_mae']:.4f} | {row['mean_lead_f1']:.4f} |"
        )
    lines += [
        "",
        "Mean test RMSE change against persistence (negative favors the listed ML model; descriptive, with no uncertainty interval):",
        "",
        "| Target | Model | Mean lead RMSE | Persistence RMSE | Difference |",
        "|---|---|---:|---:|---:|",
    ]
    for row in persistence_comparisons:
        lines.append(
            f"| {TARGET_LABELS[row['target']]} | {row['model']} | {row['mean_lead_rmse']:.4f} | {row['persistence_mean_lead_rmse']:.4f} | {row['delta_rmse_vs_persistence']:+.4f} |"
        )
    lines += [
        "",
        "## Persistence comparison and severe-drought detection",
        "",
        "See `baseline_metrics.csv` for every model, feature variant, scale, lead and pooled/per-site breakdown. The severe category uses -2.0 < SPEI ≤ -1.5, excluding the separately reported extreme category. The additional severe-or-extreme event threshold is SPEI ≤ -1.5. Pooled four-lead mean severe-category scores are:",
        "",
        "| Model | Variant | Target | Severe precision | Severe recall | Severe-or-extreme event recall |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in severe_summary.loc[severe_summary["feature_variant"] == "all_features"].to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['feature_variant']} | {row['target_column']} | {row['mean_lead_severe_precision']:.4f} | {row['mean_lead_severe_recall']:.4f} | {row['mean_lead_severe_or_extreme_recall']:.4f} |"
        )
    lines += [
        "",
        "Class counts and full seven-class confusion matrices are also saved. No model is declared superior on a small test-score difference; site and lead variation should be considered.",
        "",
        "## Sites with largest observed RMSE spread",
        "",
        "| Model | Variant | Target | Mean site RMSE | Maximum site RMSE | Minimum site RMSE |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in weak.to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['feature_variant']} | {row['target_column']} | {row['site_rmse_mean']:.4f} | {row['site_rmse_max']:.4f} | {row['site_rmse_min']:.4f} |"
        )
    lines += [
        "",
        "## Leakage, sensitivity and limitations",
        "",
        f"Window and index-availability checks: `{json.dumps(windows_audit, sort_keys=True)}`. Hyperparameter selection used validation only; test evaluation was run once after selection and is now consumed for final comparison. The no-index ablation results are in `baseline_sensitivity_analysis.csv`.",
        "",
        "These results are temporal forecasts at the 12 represented locations, not spatial transfer. The targets remain locally computed empirical SPEI with the Phase 1 limitations; this baseline phase does not establish label validity or publication readiness. Single-seed tree baselines, overlapping temporal windows, correlated sites and limited independent drought episodes constrain uncertainty claims. The report supports benchmark comparison, not causal interpretation or operational skill.",
        "",
        "## Reproduction and artifacts",
        "",
        "Run `python -m experiments.run_phase2_1_baselines --config configs/config.yaml --seed 42`. This command refuses to overwrite an existing run directory.",
        "",
        "Machine-readable outputs in this run directory: `baseline_metrics.csv`, `baseline_per_site_metrics.csv`, `baseline_hyperparameters.json`, `baseline_experiment_metadata.json`, `baseline_confusion_matrices.json`, `baseline_class_distribution.csv`, `baseline_leakage_audit.json`, and `baseline_sensitivity_analysis.csv`. Selected fitted estimators are under the matching `models/baselines/<run_id>/` directory.",
        "",
        "Tests and Ruff results are recorded in the execution summary accompanying this report.",
    ]
    (run_output_dir / "baseline_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "run_id": run_id,
        "output_dir": str(run_output_dir),
        "models_dir": str(run_models_dir),
        "test_evaluation_consumed": True,
        "xgboost_status": metadata["xgboost_status"],
        "xgboost_reason": xgb_error,
        "best_scale_summary": best.to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("results/maharashtra/phase2/baselines"))
    parser.add_argument("--model-root", type=Path, default=Path("models/baselines"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_root, args.model_root, args.seed), indent=2))


if __name__ == "__main__":
    main()
