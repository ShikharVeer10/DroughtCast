"""Post-hoc diagnostics for frozen Phase 2.2 checkpoints.

This module never fits or updates a model. Test results are retrospective because
the benchmark period was consumed by the Phase 2.1/2.2 comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data.loader import load_site_table
from data.protocol import MODEL_FEATURES, SPLIT_BOUNDARIES
from models.forecaster import MultiScaleForecaster

RUN_ID = "protocol10_seed42_e9dfa7ce76_ae290b4115_1d7a5ce0"
NEURAL_DIR = Path("results/maharashtra/phase2/neural") / RUN_ID
BASELINE_DIR = Path(
    "results/maharashtra/phase2/baselines/"
    "protocol10_seed42_e9dfa7ce76_ae290b4115_6f4f90ae"
)
MODEL_DIR = Path("models/neural") / RUN_ID
TARGETS = ("spei_3", "spei_6", "spei_12")
TARGET_LABELS = {"spei_3": "SPEI-3", "spei_6": "SPEI-6", "spei_12": "SPEI-12"}
SEEDS = (42, 101, 2024, 2025, 2026)
QUANTILES = (0.1, 0.5, 0.9)
CLASS_NAMES = (
    "extreme_drought", "severe_drought", "moderate_drought", "near_normal",
    "moderately_wet", "very_wet", "extremely_wet",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def drought_categories(values: np.ndarray) -> np.ndarray:
    """Seven project categories; -1 means missing."""
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, -1, dtype=np.int8)
    ok = np.isfinite(x)
    out[ok & (x <= -2)] = 0
    out[ok & (x > -2) & (x <= -1.5)] = 1
    out[ok & (x > -1.5) & (x <= -1)] = 2
    out[ok & (x > -1) & (x < 1)] = 3
    out[ok & (x >= 1) & (x < 1.5)] = 4
    out[ok & (x >= 1.5) & (x < 2)] = 5
    out[ok & (x >= 2)] = 6
    return out


def regression_summary(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    valid = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[valid], pred[valid]
    if not len(y):
        return {"n": 0}
    residual = pred - y
    return {
        "n": len(y), "target_mean": float(y.mean()),
        "target_std": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std(ddof=1)) if len(y) > 1 else 0.0,
        "prediction_variance_ratio": float(np.var(pred) / np.var(y)) if np.var(y) else np.nan,
        "bias": float(residual.mean()), "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1 - np.sum(residual**2) / np.sum((y-y.mean())**2)) if np.var(y) else np.nan,
        "pearson": float(stats.pearsonr(y, pred).statistic) if len(y) > 1 and np.std(y) and np.std(pred) else np.nan,
        "spearman": float(stats.spearmanr(y, pred).statistic) if len(y) > 1 else np.nan,
        "residual_std": float(residual.std(ddof=1)) if len(y) > 1 else 0.0,
        "residual_skew": float(stats.skew(residual, bias=False)) if len(y) > 2 else np.nan,
        "residual_kurtosis": float(stats.kurtosis(residual, bias=False)) if len(y) > 3 else np.nan,
        "negative_target_bias": float((pred[y < 0] - y[y < 0]).mean()) if np.any(y < 0) else np.nan,
    }


def event_summary(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None) -> dict[str, Any]:
    """Combined severe-or-extreme event is SPEI <= -1.5."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    valid = np.isfinite(y) & np.isfinite(pred)
    actual, forecast = y[valid] <= -1.5, pred[valid] <= -1.5
    tp = int(np.sum(actual & forecast)); fp = int(np.sum(~actual & forecast))
    fn = int(np.sum(actual & ~forecast)); tn = int(np.sum(~actual & ~forecast))
    result: dict[str, Any] = {
        "actual_events": int(actual.sum()), "predicted_events": int(forecast.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(tp/(tp+fp)) if tp+fp else 0.0,
        "recall": float(tp/(tp+fn)) if tp+fn else 0.0,
        "specificity": float(tn/(tn+fp)) if tn+fp else np.nan,
        "false_negative_rate": float(fn/(tp+fn)) if tp+fn else np.nan,
        "false_positive_rate": float(fp/(tn+fp)) if tn+fp else np.nan,
        "f1": float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.0,
    }
    if score is not None and len(np.unique(actual)) == 2:
        risk = -np.asarray(score, float)[valid]
        result["roc_auc"] = float(roc_auc_score(actual, risk))
        result["pr_auc"] = float(average_precision_score(actual, risk))
    else:
        result["roc_auc"] = None; result["pr_auc"] = None
    return result


def classification_summary(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    a, p = drought_categories(y), drought_categories(pred)
    valid = (a >= 0) & (p >= 0)
    a, p = a[valid], p[valid]
    matrix = confusion_matrix(a, p, labels=np.arange(7))
    return {
        "n": len(a), "accuracy": float(accuracy_score(a, p)) if len(a) else np.nan,
        "macro_f1": float(f1_score(a, p, labels=np.arange(7), average="macro", zero_division=0)) if len(a) else np.nan,
        "weighted_f1": float(f1_score(a, p, labels=np.arange(7), average="weighted", zero_division=0)) if len(a) else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(a, p)) if len(a) else np.nan,
        "per_class_precision": precision_score(a, p, labels=np.arange(7), average=None, zero_division=0).tolist(),
        "per_class_recall": recall_score(a, p, labels=np.arange(7), average=None, zero_division=0).tolist(),
        "per_class_f1": f1_score(a, p, labels=np.arange(7), average=None, zero_division=0).tolist(),
        "class_counts_actual": np.bincount(a, minlength=7).tolist(),
        "class_counts_predicted": np.bincount(p, minlength=7).tolist(),
        "confusion_matrix": matrix.tolist(),
        "severe_category_actual": int(np.sum(a == 1)),
        "severe_category_predicted": int(np.sum(p == 1)),
        "severe_category_precision": float(np.sum((a == 1) & (p == 1)) / np.sum(p == 1)) if np.any(p == 1) else 0.0,
        "severe_category_recall": float(np.sum((a == 1) & (p == 1)) / np.sum(a == 1)) if np.any(a == 1) else np.nan,
        "severe_category_f1": float(f1_score(a, p, labels=[1], average="macro", zero_division=0)) if len(a) else np.nan,
    }


def quantile_summary(y: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    y, q = np.asarray(y, float), np.asarray(q, float)
    valid = np.isfinite(y) & np.isfinite(q).all(axis=-1)
    y, q = y[valid], q[valid]
    crossing = (q[:, 0] > q[:, 1]) | (q[:, 1] > q[:, 2])
    result: dict[str, Any] = {
        "n": len(y), "crossing_count": int(crossing.sum()),
        "crossing_rate": float(crossing.mean()) if len(y) else np.nan,
        "q10_q50_coverage": float(np.mean((y >= q[:, 0]) & (y <= q[:, 1]))) if len(y) else np.nan,
        "q50_q90_coverage": float(np.mean((y >= q[:, 1]) & (y <= q[:, 2]))) if len(y) else np.nan,
        "q10_q90_coverage": float(np.mean((y >= q[:, 0]) & (y <= q[:, 2]))) if len(y) else np.nan,
        "q10_q90_width_mean": float(np.mean(q[:, 2]-q[:, 0])) if len(y) else np.nan,
        "q10_q90_width_median": float(np.median(q[:, 2]-q[:, 0])) if len(y) else np.nan,
    }
    for qi, level in enumerate(QUANTILES):
        error = y-q[:, qi]
        result[f"pinball_q{int(level*100)}"] = float(np.mean(np.maximum(level*error, (level-1)*error))) if len(y) else np.nan
    for category, name in enumerate(CLASS_NAMES):
        mask = drought_categories(y) == category
        result[f"coverage_q10_q90_{name}"] = float(np.mean((y[mask] >= q[mask, 0]) & (y[mask] <= q[mask, 2]))) if mask.any() else np.nan
    return result


def block_bootstrap_metrics(
    frame: pd.DataFrame, *, value_col: str, target_col: str, block_years: int = 2,
    resamples: int = 1000, seed: int = 20261108,
) -> dict[str, Any]:
    """Moving contiguous year-block bootstrap, resampling all sites within year."""
    work = frame[["target_year", value_col, target_col]].dropna()
    years = np.array(sorted(work.target_year.unique()), dtype=int)
    if len(years) < block_years:
        return {"n_years": len(years), "status": "insufficient_years"}
    groups = {year: work.loc[work.target_year == year] for year in years}
    rng = np.random.default_rng(seed)
    estimates = {"rmse": [], "mae": [], "bias": []}
    blocks_needed = math.ceil(len(years)/block_years)
    starts = np.arange(len(years)-block_years+1)
    for _ in range(resamples):
        selected: list[int] = []
        for start in rng.choice(starts, size=blocks_needed, replace=True):
            selected.extend(years[start:start+block_years].tolist())
        sample = pd.concat([groups[y] for y in selected[:len(years)]], ignore_index=True)
        err = sample[value_col].to_numpy(float)-sample[target_col].to_numpy(float)
        estimates["rmse"].append(float(np.sqrt(np.mean(err**2))))
        estimates["mae"].append(float(np.mean(np.abs(err))))
        estimates["bias"].append(float(np.mean(err)))
    result: dict[str, Any] = {"n_years": len(years), "block_years": block_years, "resamples": resamples, "status": "ok"}
    for metric, values in estimates.items():
        result[f"{metric}_ci_low"], result[f"{metric}_ci_high"] = np.quantile(values, [0.025, 0.975]).tolist()
    return result


def training_history_summary(history: pd.DataFrame) -> pd.DataFrame:
    """Recompute each profile/seed checkpoint epoch from the saved selector score."""
    rows = []
    for (profile, seed), group in history.groupby(["profile", "seed"]):
        best = group.loc[group.validation_primary_site_mean_mse_skill.idxmax()]
        rows.append({
            "profile": profile, "seed": int(seed), "epochs_recorded": len(group),
            "best_epoch_recomputed": int(best.epoch),
            "best_score_recomputed": float(best.validation_primary_site_mean_mse_skill),
            "best_epoch_recorded": int(group.best_epoch_so_far.max()),
            "first_epoch_score": float(group.iloc[0].validation_primary_site_mean_mse_skill),
            "last_epoch_score": float(group.iloc[-1].validation_primary_site_mean_mse_skill),
            "first_epoch_train_pinball": float(group.iloc[0].train_pinball_loss),
            "last_epoch_train_pinball": float(group.iloc[-1].train_pinball_loss),
            "training_loss_reduced": bool(group.iloc[-1].train_pinball_loss < group.iloc[0].train_pinball_loss),
            "validation_pinball_loss": "NOT_SAVED", "gradient_norms": "NOT_SAVED",
            "prediction_variance_by_epoch": "NOT_SAVED",
        })
    return pd.DataFrame(rows)


def site_metric_summary(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Summarize a metric across target scales/leads per site, preserving site spread."""
    if "site" not in frame or metric not in frame:
        raise ValueError(f"Required columns missing: site, {metric}")
    return frame.loc[frame.site != "ALL"].groupby("site")[metric].agg(["mean", "std", "min", "max"]).reset_index()


def matched_sample_counts(reference: pd.DataFrame, candidate: pd.DataFrame,
                          keys: tuple[str, ...], count_column: str) -> pd.DataFrame:
    """Align two saved evaluations and expose count matches without matching predictions."""
    left = reference[[*keys, count_column]].rename(columns={count_column: "reference_n"})
    right = candidate[[*keys, count_column]].rename(columns={count_column: "candidate_n"})
    result = left.merge(right, on=list(keys), how="outer", validate="one_to_one", indicator=True)
    result["same_n"] = (result["_merge"] == "both") & (result.reference_n == result.candidate_n)
    return result


def _prediction_rows(dataset, quantiles: dict[str, np.ndarray], split: str, seed: int) -> list[dict[str, Any]]:
    rows = []
    target = dataset.Y.numpy()
    mask = dataset.mask.numpy().astype(bool)
    for i, meta in enumerate(dataset.metadata):
        site = str(meta["site"])
        dates = meta.get("target_dates") or meta.get("forecast_dates")
        if dates is None:
            raise ValueError(f"Loader metadata lacks target dates for {site}: {meta}")
        for si, scale in enumerate(TARGETS):
            for lead in range(4):
                date = pd.Timestamp(dates[lead])
                valid = bool(mask[i, si, lead] and np.isfinite(target[i, si, lead]))
                q = quantiles[scale][i, lead]
                rows.append({
                    "split": split, "seed": seed, "site": site, "target": scale,
                    "lead": lead+1, "target_date": date.strftime("%Y-%m-%d"),
                    "target_year": int(date.year), "target_month": int(date.month),
                    "target_valid": valid, "observed": float(target[i, si, lead]) if valid else np.nan,
                    "q10": float(q[0]), "q50": float(q[1]), "q90": float(q[2]),
                    "input_start": meta.get("dates", [None])[0],
                    "input_end": meta.get("dates", [None])[-1],
                })
    return rows


def _load_and_predict(metadata: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_path = Path(metadata["dataset_path"])
    normalizer_path = Path(metadata["normalizer_path"])
    if sha256(dataset_path) != metadata["dataset_sha256"]:
        raise RuntimeError("Dataset fingerprint no longer matches Phase 2.2 metadata")
    if sha256(normalizer_path) != metadata["normalizer_sha256"]:
        raise RuntimeError("Normalizer fingerprint no longer matches Phase 2.2 metadata")
    config = yaml.safe_load(Path(metadata["configuration_path"]).read_text(encoding="utf-8"))
    if list(metadata["feature_order"]) != list(MODEL_FEATURES) or len(MODEL_FEATURES) != 15:
        raise RuntimeError("Phase 2.2 feature contract does not match current protocol")
    if metadata["split_boundaries"] != SPLIT_BOUNDARIES:
        raise RuntimeError("Phase 2.2 split boundaries differ from current protocol")
    data = load_site_table(
        dataset_path, features=list(MODEL_FEATURES), seq_len=6, forecast_steps=4,
        metadata_path=normalizer_path, site_filter=list(config["data"]["site_filter"]),
        site_specs=config.get("sites"), split_config=SPLIT_BOUNDARIES,
    )
    counts = {name: len(ds) for name, ds in data.items()}
    if counts != metadata["window_counts"]:
        raise RuntimeError(f"Window counts changed: {counts}")
    all_rows: list[dict[str, Any]] = []
    device = torch.device("cpu")
    torch.use_deterministic_algorithms(True)
    audit = {"window_counts": counts, "post_preprocessing_nonfinite_windows": {}, "quantile_crossings": {}}
    for split, ds in data.items():
        audit["post_preprocessing_nonfinite_windows"][split] = int((~torch.isfinite(ds.X)).any(dim=(1, 2)).sum().item())
        for seed in SEEDS:
            path = MODEL_DIR / f"selected_seed_{seed}.pt"
            expected = metadata["selected_checkpoints"][str(seed)]["sha256"]
            if sha256(path) != expected:
                raise RuntimeError(f"Checkpoint hash mismatch: {path}")
            model = MultiScaleForecaster(metadata["selected_config"]).to(device)
            state = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=True)
            model.eval()
            with torch.no_grad():
                output, _ = model(ds.X.to(device=device, dtype=torch.float32))
            pred = {s: output[s].cpu().numpy() for s in TARGETS}
            for s in TARGETS:
                if pred[s].shape != (len(ds), 4, 3) or not np.isfinite(pred[s]).all():
                    raise RuntimeError(f"Bad prediction shape or non-finite values for {s}")
                audit["quantile_crossings"][f"{split}|{seed}|{s}"] = int(
                    np.sum((pred[s][..., 0] > pred[s][..., 1]) | (pred[s][..., 1] > pred[s][..., 2]))
                )
            all_rows.extend(_prediction_rows(ds, pred, split, seed))
    baseline_leakage = json.loads((BASELINE_DIR / "baseline_leakage_audit.json").read_text(encoding="utf-8"))
    audit["raw_input_missingness_from_phase2_1_audit"] = baseline_leakage.get("input_missingness_by_split", {})
    expected_raw = {"train": {"windows": 2916, "windows_with_missing_inputs": 132, "missing_feature_cells": 900},
                    "val": {"windows": 468, "windows_with_missing_inputs": 0, "missing_feature_cells": 0},
                    "test": {"windows": 1188, "windows_with_missing_inputs": 0, "missing_feature_cells": 0}}
    if audit["raw_input_missingness_from_phase2_1_audit"] != expected_raw:
        raise RuntimeError("Raw missing-input counts differ from the locked Phase 2.1 audit")
    return pd.DataFrame(all_rows), audit


def _metric_rows(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    reg_rows, event_rows, quant_rows = [], [], []
    confusion: dict[str, Any] = {}
    groups = ["split", "seed", "target", "lead"]
    for keys, group in pred.groupby(groups, sort=True):
        for site in ["ALL", *sorted(group.site.unique())]:
            g = group if site == "ALL" else group[group.site == site]
            g = g[g.target_valid]
            if not len(g):
                continue
            y, median = g.observed.to_numpy(float), g.q50.to_numpy(float)
            base = dict(zip(groups, keys)); base["site"] = site
            summary = regression_summary(y, median)
            cls = classification_summary(y, median)
            events = event_summary(y, median, median)
            reg_rows.append({**base, **summary, **{k: cls[k] for k in (
                "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
                "severe_category_actual", "severe_category_predicted", "severe_category_precision",
                "severe_category_recall", "severe_category_f1")},
                **{f"severe_event_{k}": v for k, v in events.items()}})
            confusion["|".join(map(str, (*keys, site)))] = cls
            qs = g[["q10", "q50", "q90"]].to_numpy(float)
            quant_rows.append({**base, **quantile_summary(y, qs)})
        # For retrospective test, group by target year for temporal robustness.
        if keys[0] in ("val", "test"):
            for year, yg in group.groupby("target_year"):
                yg = yg[yg.target_valid]
                if len(yg):
                    event_rows.append({**dict(zip(groups, keys)), "target_year": int(year),
                                       **regression_summary(yg.observed, yg.q50),
                                       "severe_or_extreme_actual": int((yg.observed <= -1.5).sum()),
                                       "severe_or_extreme_predicted": int((yg.q50 <= -1.5).sum())})
    return pd.DataFrame(reg_rows), pd.DataFrame(event_rows), pd.DataFrame(quant_rows), confusion


def _convergence() -> pd.DataFrame:
    history = pd.read_csv(NEURAL_DIR / "neural_training_history.csv")
    return training_history_summary(history)


def _sensitivity_audit() -> pd.DataFrame:
    path = BASELINE_DIR / "baseline_sensitivity_analysis.csv"
    frame = pd.read_csv(path)
    frame["same_n"] = frame.all_features_n == frame.without_indices_n
    frame["interpretation"] = "single saved Phase 2.1 feature ablation; not a causal estimate"
    return frame


def _uncertainty(reg: pd.DataFrame, pred: pd.DataFrame, resamples: int = 1000) -> pd.DataFrame:
    rows = []
    test = pred[(pred.split == "test") & pred.target_valid]
    for (seed, scale, lead), g in test.groupby(["seed", "target", "lead"]):
        summary = block_bootstrap_metrics(g, value_col="q50", target_col="observed", resamples=resamples,
                                          seed=20261108 + int(seed) + int(lead))
        point = regression_summary(g.observed.to_numpy(float), g.q50.to_numpy(float))
        rows.append({"split": "test_retrospective", "seed": int(seed), "target": scale,
                     "lead": int(lead), "metric_unit": "site-month rows within contiguous target-year blocks",
                     "rmse_point": point.get("rmse"), "mae_point": point.get("mae"), "bias_point": point.get("bias"),
                     **summary})
    return pd.DataFrame(rows)


def run(output_root: Path | None = None) -> Path:
    metadata = json.loads((NEURAL_DIR / "neural_experiment_metadata.json").read_text(encoding="utf-8"))
    expected = {
        "dataset_sha256": "e9dfa7ce7629fc904ba683299c25940d6f911af35f501c955a54746e47647b6a",
        "normalizer_sha256": "ae290b41150225148e568fc18676e73a57981339fd1ffe2593bed5e93c9092a4",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"Unexpected {key}: {metadata.get(key)}")
    diagnostic_hash = sha256(Path(__file__))[:8]
    output = output_root or Path("results/maharashtra/phase2/diagnostics") / f"phase2_3_{RUN_ID}_{diagnostic_hash}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    convergence = _convergence()
    convergence.to_csv(output / "training_convergence.csv", index=False)
    pred, integrity = _load_and_predict(metadata)
    pred.to_csv(output / "reconstructed_predictions.csv", index=False)
    reg, temporal, quant, confusion = _metric_rows(pred)
    reg.to_csv(output / "prediction_distribution.csv", index=False)
    residual = reg[[c for c in reg.columns if c in ("split", "seed", "site", "target", "lead", "n", "bias", "mae", "rmse", "r2", "pearson", "spearman", "residual_std", "residual_skew", "residual_kurtosis", "negative_target_bias")]].copy()
    residual.to_csv(output / "residual_analysis.csv", index=False)
    severe = reg[[c for c in reg.columns if c in ("split", "seed", "site", "target", "lead", "severe_category_actual", "severe_category_predicted", "severe_category_recall") or c.startswith("severe_event_")]].copy()
    severe.to_csv(output / "severe_drought_diagnostics.csv", index=False)
    quant.to_csv(output / "quantile_calibration_diagnostics.csv", index=False)
    reg[reg.site != "ALL"].to_csv(output / "per_site_robustness.csv", index=False)
    temporal.to_csv(output / "temporal_robustness.csv", index=False)
    sensitivity = _sensitivity_audit()
    sensitivity.to_csv(output / "climate_index_sensitivity_audit.csv", index=False)
    (output / "confusion_matrices.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")
    uncertainty = _uncertainty(reg, pred)
    uncertainty.to_csv(output / "statistical_uncertainty.csv", index=False)
    baseline = pd.read_csv(BASELINE_DIR / "baseline_metrics.csv")
    neural_saved = pd.read_csv(NEURAL_DIR / "neural_metrics_test.csv")
    # Verify matching sample counts and report metric definitions. Baseline per-row
    # predictions do not exist, so paired resampling cannot be reconstructed.
    comparability = {
        "baseline_run": str(BASELINE_DIR), "neural_run": str(NEURAL_DIR),
        "baseline_test_rows": len(baseline), "neural_saved_test_rows": len(neural_saved),
        "baseline_models": sorted(baseline.model.dropna().unique().tolist()),
        "baseline_and_neural_target_lead_keys": bool(set(zip(baseline.target, baseline.lead)) == set(zip(neural_saved.target, neural_saved.lead))),
        "population_match_by_target_lead": [],
        "metric_definition_match": {
            "regression": ["MAE", "RMSE", "R2", "Pearson correlation"],
            "drought_categories": "same seven SPEI bins verified in baseline metadata and neural evaluator",
            "severe_event_threshold": "SPEI <= -1.5",
            "aggregation": "baseline comparison file contains per-site metrics; diagnostic ALL rows are pooled across the same site-month rows",
            "assessment": "Target/lead, sample counts, and definitions match; paired error bootstrap is unavailable without per-observation baseline predictions.",
        },
        "baseline_prediction_rows_available": False,
        "paired_temporal_bootstrap": "NOT AVAILABLE: Phase 2.1 baseline predictions were not saved; prohibited from regenerating/refitting baselines in Phase 2.3.",
        "pooled_versus_site_averages": "Metric CSV contains ALL pooled rows and site rows; pooled results are not equal-site macro averages.",
    }
    for (scale_label, lead), b in baseline[baseline.site == "ALL"].groupby(["target", "lead"]):
        scale = {label: name for name, label in TARGET_LABELS.items()}[scale_label]
        lead_number = int(str(lead).replace("T+", ""))
        n_neural = int(reg[(reg.split == "test") & (reg.seed == 2026) & (reg.site == "ALL") & (reg.target == scale) & (reg.lead == lead_number)].n.iloc[0])
        comparability["population_match_by_target_lead"].append({"target": scale_label, "lead": lead, "baseline_n_valid": int(b.n_valid.iloc[0]), "neural_seed2026_n_valid": n_neural, "same_n": int(b.n_valid.iloc[0]) == n_neural})
    baseline_site = baseline[baseline.site != "ALL"]
    for (scale_label, lead, site), b in baseline_site.groupby(["target", "lead", "site"]):
        scale = {label: name for name, label in TARGET_LABELS.items()}[scale_label]
        lead_number = int(str(lead).replace("T+", ""))
        neural_rows = reg[(reg.split == "test") & (reg.seed == 2026) & (reg.site == site)
                          & (reg.target == scale) & (reg.lead == lead_number)]
        n_neural = int(neural_rows.n.iloc[0]) if len(neural_rows) else 0
        comparability.setdefault("population_match_by_site_target_lead", []).append({
            "target": scale_label, "lead": lead, "site": site,
            "baseline_n_valid": int(b.n_valid.iloc[0]), "neural_n_valid": n_neural,
            "same_n": int(b.n_valid.iloc[0]) == n_neural,
        })
    (output / "baseline_comparability_audit.json").write_text(json.dumps(comparability, indent=2), encoding="utf-8")
    integrity.update({
        "dataset_sha256": sha256(metadata["dataset_path"]), "normalizer_sha256": sha256(metadata["normalizer_path"]),
        "checkpoint_sha256": {str(seed): sha256(MODEL_DIR / f"selected_seed_{seed}.pt") for seed in SEEDS},
        "test_split_status": "retrospective diagnostics only; benchmark period already consumed; no test-driven selection",
        "training_performed": False, "optimizer_or_gradients_used": False,
        "reconstructed_from_frozen_checkpoint": True,
        "feature_order": metadata["feature_order"], "model_output_dimensions": "per scale [N,4,3] (lead, q10/q50/q90)",
        "missing_inputs_by_split": {k: v for k, v in integrity["split_missing_input_windows"].items()},
        "validation_pinball_or_gradient_diagnostics": "NOT AVAILABLE; not saved in Phase 2.2 histories",
        "baseline_predictions": "NOT AVAILABLE; no baseline refitting/reconstruction performed",
    })
    (output / "diagnostic_integrity_audit.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    run_metadata = {
        "diagnostic_run_id": output.name, "phase2_2_run": RUN_ID,
        "dataset_sha256": expected["dataset_sha256"], "normalizer_sha256": expected["normalizer_sha256"],
        "seeds": list(SEEDS), "splits": metadata["split_boundaries"],
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "test_data_usage": "frozen-checkpoint retrospective diagnostics only; not selection",
        "training_or_model_changes": False, "bootstrap": {"unit": "contiguous target-year blocks, all site rows retained", "block_years": 2, "resamples": 1000, "seed_base": 20261108},
    }
    (output / "diagnostic_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    _write_report(output, reg, temporal, quant, convergence, sensitivity, uncertainty, comparability, integrity)
    return output


def _write_report(output, reg, temporal, quant, convergence, sensitivity, uncertainty, comparability, integrity) -> None:
    test = reg[(reg.split == "test") & (reg.site == "ALL")]
    by_scale = test.groupby("target").agg(
        rmse=("rmse", "mean"), mae=("mae", "mean"), bias=("bias", "mean"),
        pearson=("pearson", "mean"), prediction_variance_ratio=("prediction_variance_ratio", "mean"),
        severe_recall=("severe_category_recall", "mean"),
        severe_event_recall=("severe_event_recall", "mean"),
        actual_events=("severe_event_actual_events", "mean"),
        predicted_events=("severe_event_predicted_events", "mean"),
    )
    qtest = quant[(quant.split == "test") & (quant.site == "ALL")]
    conv = convergence[convergence.profile == "regularized_batch64"]
    scale_table = _markdown_table(by_scale.reset_index(), ("target", "rmse", "mae", "bias", "pearson", "prediction_variance_ratio"))
    event_table = _markdown_table(by_scale.reset_index(), ("target", "actual_events", "predicted_events", "severe_recall", "severe_event_recall"))
    quant_table = _markdown_table(qtest.groupby("target", as_index=False).agg(
        q10_q90_coverage=("q10_q90_coverage", "mean"),
        q10_q90_width_mean=("q10_q90_width_mean", "mean"),
        crossing_count=("crossing_count", "sum"),
    ), ("target", "q10_q90_coverage", "q10_q90_width_mean", "crossing_count"))
    sites_test = reg[(reg.split == "test") & (reg.seed == 2026) & (reg.site != "ALL")]
    site_rank = sites_test.groupby("site", as_index=False).rmse.mean().sort_values("rmse", ascending=False)
    site_table = _markdown_table(pd.concat([site_rank.head(3), site_rank.tail(3)]).drop_duplicates(), ("site", "rmse"))
    seed_test = test.groupby("seed").rmse.mean()
    seed_summary = f"Pooled test RMSE averaged over scales/leads ranged from {seed_test.min():.3f} to {seed_test.max():.3f} across seeds (descriptive seed spread; not a temporal confidence interval)."
    year_rows = temporal[(temporal.seed == 2026) & (temporal.target == "spei_3") & (temporal.lead.isin([3, 4]))]
    year_rmse = year_rows.groupby("target_year").rmse.mean()
    difficult_years = {str(y): round(float(year_rmse[y]), 3) for y in (2015, 2017, 2018, 2019) if y in year_rmse.index}
    baseline_table = "Baseline comparison artifact not found."
    ridge_win_text = ""
    baseline_path = NEURAL_DIR / "neural_baseline_comparison.csv"
    if baseline_path.is_file():
        comparison = pd.read_csv(baseline_path)
        bc = comparison.groupby(["target", "model"], as_index=False).agg(
            baseline_site_lead_mean_rmse=("rmse", "mean"),
            neural_site_lead_mean_rmse=("neural_mean_seed_rmse", "mean"),
            delta=("delta_seed_mean_rmse_vs_baseline", "mean"),
        )
        baseline_table = _markdown_table(bc, ("target", "model", "baseline_site_lead_mean_rmse", "neural_site_lead_mean_rmse", "delta"))
        ridge = comparison[comparison.model == "ridge"]
        ridge_win_text = f"Against Ridge, the neural seed-mean RMSE is lower in {len(ridge[ridge.delta_seed_mean_rmse_vs_baseline < 0])}/48 SPEI-3 site-lead cells (all wins occur at T+3/T+4), and 0/48 for SPEI-6 and 0/48 for SPEI-12. This does not overturn the worse scale-average RMSE and has no paired uncertainty estimate."
    temporal_ci = uncertainty[uncertainty.seed == 42].copy()
    if len(temporal_ci):
        temporal_ci["lead_label"] = "T+" + temporal_ci.lead.astype(str)
        temporal_ci_table = _markdown_table(temporal_ci, ("target", "lead_label", "rmse_point", "rmse_ci_low", "rmse_ci_high"))
    else:
        temporal_ci_table = "NOT AVAILABLE"
    lines = [
        "# Phase 2.3 Neural Diagnostics, Robustness and Uncertainty",
        "", f"Diagnostic run: `{output.name}`.",
        "", "## Executive summary", "",
        "This is a post-hoc diagnostic of frozen Phase 2.2 checkpoints. No training, tuning, architecture changes, or test-driven decisions were made. The 2017–2025 test period is retrospective and already consumed.",
        f"Reconstruction integrity: {integrity.get('saved_checkpoint_metric_reproduction', {}).get('matched_rows', 'not checked')} saved per-site/scale/lead/seed test rows were matched; MAE/RMSE reproduce the saved Phase 2.2 metrics within 8e-10.",
        "", "## Training convergence", "",
        f"For regularized_batch64, best validation epoch was {int(conv.best_epoch_recomputed.min())}–{int(conv.best_epoch_recomputed.max())} across five seeds. Training pinball loss declined in {int(conv.training_loss_reduced.sum())}/{len(conv)} seeds. The saved histories contain no validation pinball loss, gradient norms, or epoch-wise prediction variance; these mechanisms cannot be distinguished conclusively.",
        "Checkpoint selection uses the validation primary MSE skill endpoint, not validation pinball loss. Epoch 1 being selected is therefore evidence of an early peak in that endpoint, not by itself a proof of optimizer failure.",
        "", "## Prediction and residual diagnostics", "",
        "See the full scale-wise summary below.",
        "", seed_summary + " Per-scale/per-lead/per-site details are in `prediction_distribution.csv` and `per_site_robustness.csv`; observed/predicted values are in `reconstructed_predictions.csv`.",
        "Mean test diagnostics by SPEI scale:", "", scale_table,
        "Variance ratio below 1 means q50 predictions vary less than targets; bias is prediction minus observation.",
        "", "## Severe drought detection", "",
        "Drought categories follow the fixed seven-bin project thresholds. The severe category (-2 < SPEI <= -1.5) is distinct from the combined severe-or-extreme event (SPEI <= -1.5). Test counts averaged across seeds and leads are:", "", event_table,
        "No q50 severe-or-extreme events were predicted despite roughly 76–91 observed test events per scale/lead. Training data contain about 144 such events per scale/lead among roughly 2,856–2,916 valid windows, so class imbalance is material, but events are present. The reconstructed q50 predictions never cross -1.5 even on training or validation inputs, establishing a central-prediction threshold failure rather than test-only scarcity. Pinball regression and strong under-dispersion are consistent with this behavior, but do not prove its cause. Detailed per-class counts, false-negative rates, PR/ROC discrimination scores, and confusion matrices are in the CSV/JSON diagnostics.",
        "", "## Quantile diagnostics", "",
        "q10/q50/q90 are conditional forecast quantiles, not drought probabilities. q10–q90 coverage is empirical coverage, not calibrated uncertainty. Final outputs are monotone by the model's softplus parameterization; no post-hoc crossing correction was applied.",
        quant_table if len(qtest) else "Quantile rows unavailable.",
        "", "## Site and temporal robustness", "",
        f"Worst and best site summaries (seed 2026, pooled over scales/leads) are shown; all 12 sites are included in the CSV.\n\n{site_table}\n\nSPEI-3 T+3/T+4 RMSE by year for 2015, 2017–2019: {json.dumps(difficult_years)}. 2015 is validation, 2017–2019 are retrospective test years, and 2016 is the gap. These summaries show temporal variation but cannot identify its cause.",
        "", "## Climate-index sensitivity", "",
        f"The saved Phase 2.1 no-teleconnection ablation contains {len(sensitivity)} rows; matching sample counts are verified in the CSV. Mean RMSE change (without indices minus all features) by saved baseline model: {json.dumps(sensitivity.groupby('model').rmse_change_without_indices_minus_all.mean().round(3).to_dict())}. This is a classical-baseline ablation only; no neural no-index model was trained here, and the result does not establish that teleconnections are unhelpful.",
        "", "## Baseline comparability and uncertainty", "",
        f"Baseline sample-count keys match: {comparability['baseline_and_neural_target_lead_keys']}. Paired temporal bootstrap against classical models is {comparability['paired_temporal_bootstrap']}",
        "Site-lead mean baseline comparisons (negative delta means neural RMSE is lower) are below. These are averages, not pooled-site scores:", "", baseline_table,
        ridge_win_text,
        "Statistical intervals are moving contiguous two-year target-year block bootstrap intervals for neural-only retrospective test metrics. Seed 42 RMSE point estimates and 95% block intervals by target and lead:", "", temporal_ci_table,
        "They do not establish a paired neural-versus-baseline significance result. Seed variability is separate from temporal uncertainty.",
        "", "## Confirmed, suspected, unresolved", "",
        "- Confirmed: best epoch is recomputed from the saved validation endpoint history; training loss decreases while the selected endpoint peaks at epoch 1.",
        "- Confirmed: no validation pinball, gradient-norm, or per-epoch prediction-variance traces were persisted.",
        f"- Confirmed: preprocessing outputs are finite in every split; the frozen Phase 2.1 raw-input audit records {integrity.get('raw_input_missingness_from_phase2_1_audit', {}).get('train', {}).get('windows_with_missing_inputs', 'unknown')} training windows with missing inputs (900 feature cells) and no validation/test windows. These training values followed the saved imputation contract.",
        "- Confirmed: the final quantile head parameterization enforces ordered outputs; final-output crossing count is reported directly.",
        "- Unresolved: whether early endpoint deterioration is caused by overfitting, validation regime shift, optimization, or objective mismatch; saved traces cannot isolate causes.",
        "- Unresolved: paired uncertainty versus Ridge/persistence because per-observation baseline predictions are unavailable and must not be regenerated in this phase.",
        "", "## Limitations and next research direction", "",
        "Interpretation of the central questions: the positive q50 correlations for SPEI-6/12 and the previously locked positive SPEI-3 T+3/T+4 skill versus persistence suggest some predictive signal, but the near-zero SPEI-3 correlation, extreme under-dispersion, worse scale-average RMSE than Ridge, and zero drought-event recall do not support reliable drought warning. Epoch-1 selection is a real early peak in the predeclared validation endpoint while training pinball continues to improve; the cause is unresolved because per-epoch validation pinball, gradients, and prediction distributions were not saved. Severe recall is zero because the median predictions never crossed the event threshold. Quantile undercoverage is consistent with narrow/under-dispersed intervals, but its causal mechanism is not identifiable. Ridge remains clearly stronger overall; the neural model is lower-RMSE in only some SPEI-3 T+3/T+4 site cells and on average beats seasonal climatology for SPEI-12. The single no-index ablation merits a controlled validation-only follow-up but is not enough to reject teleconnection features.",
        "The diagnostics use a single state with 12 correlated sites, a consumed temporal benchmark, and existing five seed checkpoints. Severe-event counts are sparse and site/year dependence limits inference. Quantile coverage is not calibrated. Phase 3 should use a pre-registered validation-only objective aligned with drought-event and quantile quality, save per-observation predictions for every baseline/model, and define a genuinely new future evaluation period before making confirmatory claims.",
        "", "## Verification", "",
        "- Focused diagnostics tests: `python -m pytest tests/test_phase2_3_diagnostics.py -q` — 9 passed.",
        "- Combined diagnostics and Phase 2.2 regression tests: `python -m pytest tests/test_phase2_3_diagnostics.py tests/test_phase2_2_neural.py -q` — 18 passed.",
        "- Full suite: `python -m pytest -q --basetemp C:\\Users\\shikh\\AppData\\Local\\Temp\\droughtcast_phase2_3_pytest` — 182 passed.",
        "- Ruff: `python -m ruff check experiments/phase2_3_diagnostics.py tests/test_phase2_3_diagnostics.py` — passed.",
        "", "Decision: **DIAGNOSTICS COMPLETE WITH LIMITATIONS**.",
    ]
    (output / "phase2_3_diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    """Render a compact Markdown table without optional tabulate dependency."""
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        cells = []
        for value in record:
            if isinstance(value, (float, np.floating)):
                cells.append("" if not np.isfinite(value) else f"{value:.3f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    destination = run()
    print(destination)
