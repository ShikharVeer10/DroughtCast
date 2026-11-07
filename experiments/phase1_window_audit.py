"""Audit chronological windows and feature-source timing without model fitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.loader import split_for_date
from data.preprocessing import ClimateNormalizer
from data.protocol import SPLITS


def main() -> None:
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    data_path = Path(cfg["data"]["path"])
    frame = pd.read_csv(data_path, parse_dates=["date"])
    features = cfg["data"]["features"]
    seq_len = int(cfg["data"]["seq_len"])
    horizon = int(cfg["data"]["forecast_steps"])
    report = {"seq_len": seq_len, "forecast_steps": horizon, "splits": {}, "sites": {}, "checks": {"future_index_sources_in_inputs": 0, "stale_index_values_present": 0}}
    targets_by_split = {s: set() for s in ("train", "val", "test")}
    for split in targets_by_split:
        report["splits"][split] = {"windows": 0, "rejected_missing_features": 0, "rejected_missing_targets": 0, "input_start": None, "input_end": None, "target_start": None, "target_end": None}
    for site, group in frame.groupby("site", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        dates = pd.DatetimeIndex(group.date)
        split_labels = np.array([split_for_date(d) for d in dates], dtype=object)
        train_rows = group.loc[split_labels == "train", features].apply(pd.to_numeric, errors="coerce")
        norm = ClimateNormalizer().fit(train_rows.to_numpy())
        x = norm.transform(group[features].apply(pd.to_numeric, errors="coerce").to_numpy())
        y = group[["spei_3", "spei_6", "spei_12"]].to_numpy(dtype=float)
        site_report = {s: {"windows": 0, "windows_with_mean_imputation": 0, "rejected_missing_features": 0, "rejected_missing_targets": 0} for s in targets_by_split}
        for start in range(len(group) - seq_len - horizon + 1):
            ixs = np.arange(start, start + seq_len)
            tys = np.arange(start + seq_len, start + seq_len + horizon)
            labels = split_labels[np.r_[ixs, tys]]
            split = labels[0]
            if split is None or not np.all(labels == split):
                continue
            xs, ys = x[ixs], y[tys]
            raw_x = group.iloc[ixs][features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(raw_x).all():
                report["splits"][split].setdefault("windows_with_mean_imputation", 0)
                report["splits"][split]["windows_with_mean_imputation"] += 1
                site_report[split]["windows_with_mean_imputation"] += 1
            for row_idx in ixs:
                row = group.iloc[row_idx]
                input_month = dates[row_idx]
                for name in ("oni", "dmi", "nao", "pdo"):
                    source = row.get(f"{name}_source_month")
                    effective = row.get(f"{name}_effective_month")
                    if pd.notna(source) and pd.Timestamp(f"{source}-01") > input_month:
                        report["checks"]["future_index_sources_in_inputs"] += 1
                    if pd.notna(effective) and pd.Timestamp(f"{effective}-01") > input_month:
                        report["checks"]["future_index_sources_in_inputs"] += 1
                    if bool(row.get(f"{name}_is_missing", False)) and pd.notna(row.get(name)):
                        report["checks"]["stale_index_values_present"] += 1
            if not np.isfinite(xs).all():
                report["splits"][split]["rejected_missing_features"] += 1
                site_report[split]["rejected_missing_features"] += 1
                continue
            if not np.isfinite(ys).any():
                report["splits"][split]["rejected_missing_targets"] += 1
                site_report[split]["rejected_missing_targets"] += 1
                continue
            input_dates = dates[ixs]
            target_dates = dates[tys]
            expected_targets = pd.date_range(input_dates[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
            if not input_dates.equals(pd.date_range(input_dates[0], periods=seq_len, freq="MS")):
                raise RuntimeError(f"Non-consecutive input window for {site} at {input_dates[0]}")
            if not target_dates.equals(expected_targets) or target_dates[0] <= input_dates[-1]:
                raise RuntimeError(f"Invalid forecast target dates for {site} at {input_dates[-1]}")
            idx = report["splits"][split]
            idx["windows"] += 1
            site_report[split]["windows"] += 1
            targets_by_split[split].update(d.strftime("%Y-%m-%d") for d in target_dates)
            for field, val in (("input_start", input_dates[0]), ("input_end", input_dates[-1]), ("target_start", target_dates[0]), ("target_end", target_dates[-1])):
                current = idx[field]
                iso = val.strftime("%Y-%m-%d")
                idx[field] = min(current, iso) if field.endswith("start") and current else max(current, iso) if current else iso
        report["sites"][site] = site_report
    splits_disjoint = all(not (targets_by_split[a] & targets_by_split[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    report["checks"].update({"target_months_disjoint_between_splits": splits_disjoint, "chronological_inputs_and_targets": True, "dataset_period": [str(frame.date.min().date()), str(frame.date.max().date())], "split_boundaries": SPLITS, "model_feature_count": len(features), "normalizers_fit_on_train_only": True, "windows_restricted_to_one_split": True})
    if not splits_disjoint:
        raise RuntimeError("Target months overlap across splits")
    if report["checks"]["future_index_sources_in_inputs"] or report["checks"]["stale_index_values_present"]:
        raise RuntimeError("Index provenance audit found future or stale input values")
    out = Path("results/maharashtra/phase1/window_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["splits"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
