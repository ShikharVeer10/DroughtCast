"""Complete baseline classification summaries from saved confusion counts only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.protocol import INDEX_ALIGNMENT, MODEL_FEATURES, SPLIT_BOUNDARIES


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rates(matrix: list[list[int]], positive: tuple[int, ...]) -> tuple[float, float]:
    table = pd.DataFrame(matrix).to_numpy()
    selected = list(positive)
    true_positive = int(table[np.ix_(selected, selected)].sum())
    false_positive = int(table[:, selected].sum() - true_positive)
    false_negative = int(table[selected, :].sum() - true_positive)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return precision, recall


def finalize(run_dir: Path) -> dict:
    metrics_path = run_dir / "baseline_metrics.csv"
    confusion_path = run_dir / "baseline_confusion_matrices.json"
    metadata_path = run_dir / "baseline_experiment_metadata.json"
    rows = pd.read_csv(metrics_path)
    matrices = json.loads(confusion_path.read_text(encoding="utf-8"))
    exact_precision, exact_recall, event_precision, event_recall = [], [], [], []
    for row in rows.itertuples(index=False):
        key = f"{row.model}|{row.feature_variant}|{row.target_column}|{row.lead}|{row.site}"
        matrix = matrices[key]["counts"]
        p_severe, r_severe = _rates(matrix, (1,))
        p_event, r_event = _rates(matrix, (0, 1))
        exact_precision.append(p_severe)
        exact_recall.append(r_severe)
        event_precision.append(p_event)
        event_recall.append(r_event)
    rows["severe_drought_precision"] = exact_precision
    rows["severe_drought_recall"] = exact_recall
    rows["severe_or_extreme_event_precision_le_minus_1_5"] = event_precision
    rows["severe_or_extreme_event_recall_le_minus_1_5"] = event_recall
    rows.to_csv(metrics_path, index=False)
    rows.loc[rows["site"] != "ALL"].to_csv(run_dir / "baseline_per_site_metrics.csv", index=False)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw = pd.read_csv(metadata["dataset_path"], parse_dates=["date"])
    raw["site"] = raw["site"].astype(str).str.strip().str.lower()
    configured_features = metadata["features"]
    if configured_features != MODEL_FEATURES or sha256(Path(metadata["normalizer_path"])) != metadata["normalizer_sha256"]:
        raise ValueError("Baseline run feature contract or normalizer hash no longer matches its metadata")
    target_observation_checks = {}
    for scale in ("3", "6", "12"):
        obs = pd.to_numeric(raw[f"spei_{scale}_obs"], errors="coerce")
        target = pd.to_numeric(raw[f"spei_{scale}"], errors="coerce")
        mismatch = (obs.notna() != target.notna()) | (obs.notna() & target.notna() & ~np.isclose(obs, target, rtol=0, atol=1e-12))
        target_observation_checks[f"spei_{scale}_obs"] = {
            "paired_rows": int((obs.notna() & target.notna()).sum()),
            "mismatch_rows": int(mismatch.sum()),
            "matches_same_month_target": not bool(mismatch.any()),
        }

    index_checks = {}
    for name, policy in INDEX_ALIGNMENT.items():
        if not isinstance(policy, dict) or "publication_lag_months" not in policy:
            continue
        source_col = f"{name}_source_month"
        effective_col = f"{name}_effective_month"
        age_col = f"{name}_observation_age_months"
        stale_col = f"{name}_is_stale"
        missing_col = f"{name}_is_missing"
        available = raw[name].notna()
        source = pd.to_datetime(raw[source_col], errors="coerce")
        effective = pd.to_datetime(raw[effective_col], errors="coerce")
        age = pd.to_numeric(raw[age_col], errors="coerce")
        missing_source = source.isna()
        expected_effective = source + pd.offsets.MonthBegin(int(policy["publication_lag_months"]))
        wrong_lag = source.notna() & (expected_effective != effective)
        info_future = effective.notna() & (effective > pd.to_datetime(raw["date"]))
        source_future = source.notna() & (source > pd.to_datetime(raw["date"]))
        actual_age = (raw["date"].dt.year - source.dt.year) * 12 + (raw["date"].dt.month - source.dt.month)
        wrong_age = source.notna() & age.notna() & (actual_age != age)
        stale = raw[stale_col].fillna(False).astype(bool)
        missing_flag = raw[missing_col].fillna(False).astype(bool)
        if (available & stale).any() or (available & missing_flag).any():
            raise ValueError(f"Stale/missing climate-index values were silently populated: {name}")
        if ((available) & (age > int(policy["max_age_months"]))).any():
            raise ValueError(f"Over-age climate-index value was treated as usable: {name}")
        index_checks[name] = {
            "publication_lag_months": int(policy["publication_lag_months"]),
            "max_age_months": int(policy["max_age_months"]),
            "usable_values": int(available.sum()),
            "missing_values": int((~available).sum()),
            "stale_flags": int(stale.sum()),
            "missing_flags": int(missing_flag.sum()),
            "source_month_missing_count": int(missing_source.sum()),
            "publication_lag_mismatch_count": int(wrong_lag.sum()),
            "age_mismatch_count": int(wrong_age.sum()),
            "future_source_month_count": int(source_future.sum()),
            "future_effective_month_count": int(info_future.sum()),
            "all_checks_pass": not any((wrong_lag.any(), wrong_age.any(), source_future.any(), info_future.any())),
        }

    input_missingness = {split: {"windows": 0, "windows_with_missing_inputs": 0, "missing_feature_cells": 0} for split in ("train", "val", "test")}
    split_bounds = {
        "train": (None, pd.Timestamp(SPLIT_BOUNDARIES["train_end"])),
        "val": (pd.Timestamp(SPLIT_BOUNDARIES["val_start"]), pd.Timestamp(SPLIT_BOUNDARIES["val_end"])),
        "test": (pd.Timestamp(SPLIT_BOUNDARIES["test_start"]), pd.Timestamp(SPLIT_BOUNDARIES["test_end"])),
    }
    for _, group in raw.groupby("site", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        dates = pd.DatetimeIndex(group["date"])
        values = group[configured_features].replace([-999.0, -99.0, -9999.0, -32768.0], np.nan).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        targets = group[["spei_3", "spei_6", "spei_12"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        for start in range(max(0, len(group) - 6 - 4 + 1)):
            input_idx = np.arange(start, start + 6)
            target_idx = np.arange(start + 6, start + 10)
            combined = np.concatenate([input_idx, target_idx])
            split = next((name for name, (lower, upper) in split_bounds.items() if (lower is None or dates[combined[0]] >= lower) and dates[combined[-1]] <= upper), None)
            if split is None or not np.isfinite(targets[target_idx]).any():
                continue
            missing_cells = int((~np.isfinite(values[input_idx])).sum())
            input_missingness[split]["windows"] += 1
            input_missingness[split]["missing_feature_cells"] += missing_cells
            input_missingness[split]["windows_with_missing_inputs"] += int(missing_cells > 0)

    overall = rows.loc[rows["site"] == "ALL"]
    severe_summary = (
        overall.groupby(["model", "feature_variant", "target_column"], as_index=False)
        .agg(
            mean_lead_severe_precision=("severe_drought_precision", "mean"),
            mean_lead_severe_recall=("severe_drought_recall", "mean"),
            mean_lead_severe_or_extreme_recall=("severe_or_extreme_event_recall_le_minus_1_5", "mean"),
        )
    )
    report_path = run_dir / "baseline_experiment_report.md"
    report = report_path.read_text(encoding="utf-8")
    start = report.index("## Persistence comparison and severe-drought detection")
    end = report.index("## Sites with largest observed RMSE spread", start)
    lines = [
        "## Persistence comparison and severe-drought detection",
        "",
        "See `baseline_metrics.csv` for every model, feature variant, scale, lead and pooled/per-site breakdown. The exact severe category is -2.0 < SPEI ≤ -1.5; extreme drought (SPEI ≤ -2.0) is separate. A second event score combines severe and extreme drought at SPEI ≤ -1.5. Values below are pooled four-lead averages:",
        "",
        "| Model | Variant | Target | Severe precision | Severe recall | Severe-or-extreme recall |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in severe_summary.loc[severe_summary["feature_variant"] == "all_features"].to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['feature_variant']} | {row['target_column']} | {row['mean_lead_severe_precision']:.4f} | {row['mean_lead_severe_recall']:.4f} | {row['mean_lead_severe_or_extreme_recall']:.4f} |"
        )
    lines += [
        "",
        "Class counts and seven-category confusion matrices are saved as machine-readable artifacts. Test-set class metrics were derived from the already generated confusion counts; no models were refit and no test predictions were rerun.",
        "",
        "",
    ]
    report_path.write_text(report[:start] + "\n".join(lines) + report[end:], encoding="utf-8")

    metadata["severe_category_definition"] = "-2.0 < SPEI <= -1.5; extreme drought (SPEI <= -2.0) is a separate category"
    metadata["severe_or_extreme_event_definition"] = "SPEI <= -1.5"
    config_path = Path("configs/config.yaml")
    metadata["configuration_path"] = str(config_path)
    metadata["configuration_sha256"] = sha256(config_path)
    metadata["classification_metric_postprocessing"] = {
        "derived_from": str(confusion_path.name),
        "postprocessing_script_sha256": sha256(Path(__file__)),
        "models_refit": False,
        "test_predictions_recomputed": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    leakage_path = run_dir / "baseline_leakage_audit.json"
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    leakage["target_observation_feature_audit"] = target_observation_checks
    leakage["index_provenance_audit"] = index_checks
    leakage["input_missingness_by_split"] = input_missingness
    leakage["target_derived_features_are_same_month_only"] = all(
        item["matches_same_month_target"] for item in target_observation_checks.values()
    )
    leakage["no_over_age_indices_used_as_finite_inputs"] = all(
        item["all_checks_pass"] for item in index_checks.values()
    )
    leakage["missing_values_mean_imputed_from_train_normalizer"] = True
    leakage_path.write_text(json.dumps(leakage, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "metric_rows": len(rows),
        "confusion_matrices_used": len(matrices),
        "models_refit": False,
        "test_predictions_recomputed": False,
        "target_observation_feature_mismatches": sum(item["mismatch_rows"] for item in target_observation_checks.values()),
        "index_provenance_checks_pass": all(item["all_checks_pass"] for item in index_checks.values()),
        "input_missingness_by_split": input_missingness,
        "severe_category_definition": metadata["severe_category_definition"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    print(json.dumps(finalize(parser.parse_args().run_dir), indent=2))


if __name__ == "__main__":
    main()
