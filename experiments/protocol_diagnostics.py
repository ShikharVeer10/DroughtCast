"""Generate target calibration, drift, tail-range and SPEIbase diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.protocol import SPEI_CALIBRATION_END, SPLIT_BOUNDARIES


def _distribution(values: pd.Series) -> dict:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std(ddof=1)) if finite.size > 1 else None,
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "drought_events_le_minus_1": int((finite <= -1.0).sum()),
        "severe_events_le_minus_1_5": int((finite <= -1.5).sum()),
    }


def _agreement(computed: np.ndarray, reference: np.ndarray) -> dict:
    valid = np.isfinite(computed) & np.isfinite(reference)
    a, b = computed[valid], reference[valid]
    result = {"n": int(valid.sum())}
    if len(a) < 3:
        return result | {"correlation": None, "rmse": None, "bias": None}
    result.update({
        "correlation": float(np.corrcoef(a, b)[0, 1]),
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "bias": float(np.mean(a - b)),
    })
    for label, cutoff in (("drought", -1.0), ("severe", -1.5)):
        event_a, event_b = a <= cutoff, b <= cutoff
        result[f"{label}_agreement"] = float(np.mean(event_a == event_b))
        tp = int((event_a & event_b).sum())
        precision = tp / int(event_a.sum()) if event_a.sum() else None
        recall = tp / int(event_b.sum()) if event_b.sum() else None
        result[f"{label}_precision"] = precision
        result[f"{label}_recall"] = recall
        result[f"{label}_f1"] = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
    return result


def run(config_path: Path, dataset_path: Path, reference_points: Path | None) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(dataset_path, parse_dates=["date"])
    dates = frame["date"]
    splits = {
        "train": dates <= pd.Timestamp(SPLIT_BOUNDARIES["train_end"]),
        "validation": dates.between(SPLIT_BOUNDARIES["val_start"], SPLIT_BOUNDARIES["val_end"]),
        "test": dates.between(SPLIT_BOUNDARIES["test_start"], SPLIT_BOUNDARIES["test_end"]),
    }
    diagnostics: dict = {
        "config": str(config_path),
        "dataset": str(dataset_path),
        "dataset_sha256": __import__("hashlib").sha256(dataset_path.read_bytes()).hexdigest(),
        "period": [str(dates.min().date()), str(dates.max().date())],
        "sites": sorted(frame.site.astype(str).unique().tolist()),
        "split_boundaries": SPLIT_BOUNDARIES,
        "row_counts_by_split": {name: int(mask.sum()) for name, mask in splits.items()},
        "calibration_end": SPEI_CALIBRATION_END,
        "target_source": cfg["data"].get("target_source", "local"),
        "targets": {},
        "warnings": [],
    }
    for scale in (3, 6, 12):
        target = f"spei_{scale}"
        by_site = {}
        for site, group in frame.groupby("site", sort=True):
            site_dates = pd.to_datetime(group.date)
            balance = (
                pd.to_numeric(group.PRECTOTCORR, errors="coerce")
                * site_dates.dt.days_in_month.to_numpy()
                - pd.to_numeric(group.pet_mm, errors="coerce")
            )
            accum = balance.rolling(scale, min_periods=scale).sum()
            calibration = accum.loc[site_dates <= pd.Timestamp(SPEI_CALIBRATION_END)]
            monthly_counts = calibration.groupby(site_dates.loc[calibration.index].dt.month).count()
            by_site[str(site)] = {
                "calibration_count_by_calendar_month": {str(int(k)): int(v) for k, v in monthly_counts.items()},
                "calibration_count_min": int(monthly_counts.min()) if len(monthly_counts) else 0,
                "calibration_count_max": int(monthly_counts.max()) if len(monthly_counts) else 0,
                "by_split": {
                    name: _distribution(group.loc[mask.loc[group.index], target])
                    for name, mask in splits.items()
                },
            }
        split_stats = {
            name: _distribution(frame.loc[mask, target]) for name, mask in splits.items()
        }
        local_scale = {"scale": scale, "by_split": split_stats, "by_site": by_site}
        if reference_points and reference_points.is_file() and diagnostics["target_source"] == "local":
            ref = pd.read_csv(reference_points, parse_dates=["date"])
            ref_col = f"spei_{scale}_reference"
            merged = frame[["site", "date", target]].merge(
                ref[["site", "date", ref_col]], on=["site", "date"], how="inner", validate="one_to_one"
            )
            local_scale["speibase_1990_2024_pooled"] = _agreement(
                pd.to_numeric(merged[target], errors="coerce").to_numpy(),
                pd.to_numeric(merged[ref_col], errors="coerce").to_numpy(),
            )
            site_metrics = {}
            region_metrics = {}
            region_for_site = {
                name: spec.get("region", "unknown")
                for name, spec in cfg.get("sites", {}).items()
            }
            merged["region"] = merged.site.map(region_for_site)
            for site, group in merged.groupby("site", sort=True):
                site_metrics[site] = _agreement(
                    pd.to_numeric(group[target], errors="coerce").to_numpy(),
                    pd.to_numeric(group[ref_col], errors="coerce").to_numpy(),
                )
            for region, group in merged.groupby("region", sort=True):
                region_metrics[region] = _agreement(
                    pd.to_numeric(group[target], errors="coerce").to_numpy(),
                    pd.to_numeric(group[ref_col], errors="coerce").to_numpy(),
                )
            local_scale["speibase_by_site"] = site_metrics
            local_scale["speibase_by_region"] = region_metrics
        elif diagnostics["target_source"] == "speibase":
            local_scale["speibase_reference_validation"] = (
                "not applicable: official SPEIbase is the target in this dataset"
            )
        diagnostics["targets"][target] = local_scale

        minimum_calibration = min(
            item["calibration_count_min"] for item in by_site.values()
        ) if by_site else 0
        if minimum_calibration < 30:
            diagnostics["warnings"].append(
                f"{target}: some site/calendar-month calibration samples <30 (minimum={minimum_calibration}); tail estimates are uncertain."
            )
        train_mean, test_mean = split_stats["train"]["mean"], split_stats["test"]["mean"]
        train_std, test_std = split_stats["train"]["std"], split_stats["test"]["std"]
        if train_mean is not None and test_mean is not None and abs(test_mean - train_mean) > 0.2:
            diagnostics["warnings"].append(
                f"{target}: test-period pooled mean differs from train by >0.2 ({test_mean - train_mean:.3f}); inspect target distribution drift."
            )
        if train_std and test_std and abs(test_std / train_std - 1) > 0.2:
            diagnostics["warnings"].append(
                f"{target}: test/train pooled standard deviation ratio differs from 1 by >0.2 ({test_std / train_std:.3f}); inspect distribution drift."
            )
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", help="Defaults to data.path from config")
    parser.add_argument("--reference-points", default="results/maharashtra/speibase_validation/matched_monthly_points.csv")
    parser.add_argument("--output", default="results/maharashtra/protocol_diagnostics.json")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset = Path(args.dataset or cfg["data"]["path"])
    diagnostics = run(Path(args.config), dataset, Path(args.reference_points))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "warnings": diagnostics["warnings"]}, indent=2))


if __name__ == "__main__":
    main()
