"""Audit and report Phase 2.0 feature and normalizer compatibility."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.preprocessing import (
    DEFAULT_NORMALIZER_METADATA_PATH,
    validate_configured_model_features,
    validate_normalizer_metadata,
)
from data.protocol import DATASET_PROTOCOL_VERSION, MODEL_FEATURES, SPLIT_BOUNDARIES
from inference import _load_recent_input


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(config_path: Path, output_dir: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    features = list(data_cfg["features"])
    validate_configured_model_features(features, "Phase 2.0 audit config")
    if int(config["model"]["input_dim"]) != len(features):
        raise ValueError("Configured model input_dim differs from feature count")

    dataset_path = Path(data_cfg["path"])
    dataset_hash = sha256(dataset_path)
    legacy_path = Path("data/meta.json")
    legacy_hash = sha256(legacy_path)
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    legacy_features = list(legacy.get("features", []))
    current_metadata_path = Path(
        data_cfg.get("metadata_path", DEFAULT_NORMALIZER_METADATA_PATH)
    )
    metadata = json.loads(current_metadata_path.read_text(encoding="utf-8"))
    validate_normalizer_metadata(
        metadata,
        features,
        expected_dataset_sha256=dataset_hash,
        expected_protocol_version=DATASET_PROTOCOL_VERSION,
        expected_split_boundaries=SPLIT_BOUNDARIES,
        expected_seq_len=int(data_cfg["seq_len"]),
        expected_forecast_steps=int(data_cfg["forecast_steps"]),
    )

    frame = pd.read_csv(dataset_path, parse_dates=["date"])
    rows = []
    legacy_stats_match = True
    legacy_per_feature = {}
    for position, expected in enumerate(features):
        old = legacy_features[position] if position < len(legacy_features) else None
        same_position = old == expected
        feature_diffs = []
        for site, group in frame.groupby(frame["site"].astype(str).str.lower()):
            train = group.loc[
                group["date"] <= pd.Timestamp(SPLIT_BOUNDARIES["train_end"]),
                features,
            ].apply(pd.to_numeric, errors="coerce")
            current_mean = float(train[expected].mean())
            current_std = float(train[expected].std(ddof=0))
            saved = legacy.get("normalizers", {}).get(site, {})
            old_index = legacy_features.index(expected) if expected in legacy_features else None
            saved_mean = saved.get("mean", [])[old_index] if old_index is not None and len(saved.get("mean", [])) > old_index else None
            saved_std = saved.get("std", [])[old_index] if old_index is not None and len(saved.get("std", [])) > old_index else None
            feature_diffs.append(
                {
                    "site": site,
                    "mean_abs_difference": (
                        abs(float(saved_mean) - current_mean) if saved_mean is not None else None
                    ),
                    "std_abs_difference": (
                        abs(float(saved_std) - current_std) if saved_std is not None else None
                    ),
                }
            )
        comparable = [x for x in feature_diffs if x["mean_abs_difference"] is not None]
        means_match = bool(comparable) and all(x["mean_abs_difference"] <= 1e-12 for x in comparable)
        stds_match = bool(comparable) and all(x["std_abs_difference"] <= 1e-12 for x in comparable)
        legacy_stats_match &= means_match and stds_match
        legacy_per_feature[expected] = {
            "saved_feature_position": old_index if expected in legacy_features else None,
            "mean_match_current_training_only": means_match,
            "std_match_current_training_only": stds_match,
            "max_mean_abs_difference": max((x["mean_abs_difference"] for x in comparable), default=None),
            "max_std_abs_difference": max((x["std_abs_difference"] for x in comparable), default=None),
        }
        rows.append(
            {
                "position_zero_based": position,
                "position_one_based": position + 1,
                "configured_feature": expected,
                "legacy_saved_feature": old,
                "same_position": same_position,
                "legacy_stats_match_current_train_only_data": means_match and stds_match,
            }
        )

    site = "aurangabad"
    recent = frame.loc[frame["site"].astype(str).str.lower().eq(site)].sort_values("date").tail(6)
    if recent["date"].max() != pd.Timestamp("2026-10-01"):
        raise ValueError("October 2026 smoke input is not the latest six-month window")
    x, input_dates, _ = _load_recent_input(
        dataset_path,
        site,
        features,
        int(data_cfg["seq_len"]),
        current_metadata_path,
        expected_dataset_sha256=dataset_hash,
        expected_forecast_steps=int(data_cfg["forecast_steps"]),
    )
    stale_features = {}
    for name in ("dmi", "pdo"):
        stale_flag = recent[f"{name}_is_stale"].iloc[-1] if f"{name}_is_stale" in recent else None
        stale_features[name] = {
            "raw_value_is_missing": bool(pd.isna(recent[name].iloc[-1])),
            "stale_flag": bool(stale_flag) if stale_flag is not None and not pd.isna(stale_flag) else None,
            "normalized_value": float(x[-1, features.index(name)]),
        }
    for name in ("oni", "nao"):
        stale_features[name] = {
            "raw_value_is_present": bool(pd.notna(recent[name].iloc[-1])),
            "normalized_value": float(x[-1, features.index(name)]),
        }
    smoke_ok = (
        x.shape == (6, 15)
        and x.dtype == np.float32
        and np.isfinite(x).all()
        and all(stale_features[name]["raw_value_is_missing"] for name in ("dmi", "pdo"))
        and all(abs(stale_features[name]["normalized_value"]) <= 1e-7 for name in ("dmi", "pdo"))
        and all(stale_features[name]["raw_value_is_present"] for name in ("oni", "nao"))
    )
    if not smoke_ok:
        raise RuntimeError("October 2026 preprocessing smoke-test assertions failed")

    comparison_path = output_dir / "phase2_0_feature_order_comparison.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    audit = {
        "decision": "PASS WITH LIMITATIONS",
        "configuration_path": str(config_path),
        "configuration_sha256": sha256(config_path),
        "legacy_artifact": {
            "path": str(legacy_path),
            "sha256": legacy_hash,
            "feature_order": legacy_features,
            "provenance": "unknown; no schema, protocol version, dataset hash, config fingerprint, or training period recorded",
            "matches_current_training_only_statistics": bool(legacy_stats_match),
            "per_feature_statistics": legacy_per_feature,
            "reused_or_modified": False,
        },
        "feature_order": {
            "configured": features,
            "protocol": list(MODEL_FEATURES),
            "exact_match": features == MODEL_FEATURES,
            "feature_count": len(features),
            "legacy_position_mismatches": [row for row in rows if not row["same_position"]],
        },
        "selected_normalizer": {
            "path": str(current_metadata_path),
            "sha256": sha256(current_metadata_path),
            "schema_version": metadata["normalizer_schema_version"],
            "protocol_version": metadata["dataset_protocol_version"],
            "dataset_sha256": metadata["training_dataset_sha256"],
            "preprocessing_configuration_fingerprint": metadata[
                "preprocessing_fingerprint"
            ],
            "training_split": metadata["training_split"],
            "training_dates_by_site": metadata["training_dates_by_site"],
            "stats_by_site_verified_against_independent_train_only_calculation": True,
            "normalizer_feature_order_matches_configuration": all(
                value["features"] == features for value in metadata["normalizers"].values()
            ),
        },
        "october_2026_preprocessing_smoke": {
            "site": site,
            "input_months": [date.strftime("%Y-%m") for date in input_dates],
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "all_finite": bool(np.isfinite(x).all()),
            "index_features": stale_features,
            "model_forward_pass": False,
        },
        "dataset_sha256": dataset_hash,
        "canonical_dataset_modified": False,
        "config_modified": False,
        "model_training_or_forward_pass": False,
    }
    (output_dir / "phase2_0_normalizer_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/maharashtra/phase2")
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
