"""Create a separate, verified normalizer artifact from training rows only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.loader import load_site_table
from data.preprocessing import (
    DEFAULT_NORMALIZER_METADATA_PATH,
    validate_configured_model_features,
    validate_normalizer_metadata,
)
from data.protocol import DATASET_PROTOCOL_VERSION, SPLIT_BOUNDARIES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    features = list(data_config["features"])
    validate_configured_model_features(features, "Phase 2.0 config")
    if int(config["model"].get("input_dim", -1)) != len(features):
        raise ValueError("model.input_dim does not match the protocol feature count")

    dataset_path = Path(data_config["path"])
    output_path = Path(
        data_config.get("metadata_path", DEFAULT_NORMALIZER_METADATA_PATH)
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite normalizer artifact: {output_path}"
        )
    temporary_path = output_path.with_name(output_path.name + ".candidate")
    if temporary_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite temporary normalizer candidate: {temporary_path}"
        )

    dataset_hash_before = sha256(dataset_path)
    legacy_path = Path("data/meta.json")
    legacy_hash_before = sha256(legacy_path) if legacy_path.is_file() else None
    frame = pd.read_csv(dataset_path, parse_dates=["date"])
    expected_sites = sorted(data_config["site_filter"])
    if sorted(frame["site"].astype(str).str.lower().unique()) != expected_sites:
        raise ValueError("Canonical dataset site set does not match the configured 12 sites")

    datasets = load_site_table(
        dataset_path,
        features=features,
        seq_len=int(data_config["seq_len"]),
        forecast_steps=int(data_config["forecast_steps"]),
        metadata_path=temporary_path,
        site_filter=expected_sites,
        input_extras=config["model"].get("input_extras"),
        static_context=config["model"].get("static_context", False),
        residual_heads=config["model"].get("heads", "direct") == "residual",
        site_specs=config.get("sites"),
        split_config=data_config.get("split_boundaries"),
    )
    del datasets

    metadata = json.loads(temporary_path.read_text(encoding="utf-8"))
    validate_normalizer_metadata(
        metadata,
        features,
        expected_dataset_sha256=dataset_hash_before,
        expected_protocol_version=DATASET_PROTOCOL_VERSION,
        expected_split_boundaries=SPLIT_BOUNDARIES,
        expected_seq_len=int(data_config["seq_len"]),
        expected_forecast_steps=int(data_config["forecast_steps"]),
    )

    stats_checks = {}
    for site, group in frame.groupby(frame["site"].astype(str).str.lower()):
        training = group.loc[
            group["date"] <= pd.Timestamp(SPLIT_BOUNDARIES["train_end"]), features
        ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(training)
        values = np.where(finite, training, np.nan)
        expected_mean = np.nanmean(values, axis=0)
        expected_std = np.nanstd(values, axis=0)
        expected_std = np.where(np.isfinite(expected_std) & (expected_std > 0), expected_std, 1.0)
        saved = metadata["normalizers"][site]
        means_match = np.allclose(saved["mean"], expected_mean, rtol=0, atol=1e-12)
        stds_match = np.allclose(saved["std"], expected_std, rtol=0, atol=1e-12)
        stats_checks[site] = {
            "training_rows": len(training),
            "feature_order_match": saved["features"] == features,
            "all_features_have_finite_train_values": bool(finite.sum(axis=0).min() > 0),
            "mean_match_independent_training_only_recalculation": bool(means_match),
            "std_match_independent_training_only_recalculation": bool(stds_match),
            "missing_training_cells_by_feature": {
                feature: int((~finite[:, index]).sum())
                for index, feature in enumerate(features)
            },
        }
        if not (means_match and stds_match and stats_checks[site]["feature_order_match"]):
            raise RuntimeError(f"Training-only normalizer validation failed for {site}")

    if sha256(dataset_path) != dataset_hash_before:
        raise RuntimeError("Canonical dataset changed during normalizer preparation")
    if legacy_hash_before is not None and sha256(legacy_path) != legacy_hash_before:
        raise RuntimeError("Legacy normalizer artifact changed during preparation")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.replace(output_path)
    return {
        "artifact_path": str(output_path),
        "legacy_artifact_preserved": str(legacy_path),
        "legacy_artifact_sha256": legacy_hash_before,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash_before,
        "protocol_version": DATASET_PROTOCOL_VERSION,
        "features": features,
        "feature_count": len(features),
        "training_date_range": ["1990-01", "2010-12"],
        "fit_rows_per_site": 252,
        "fit_scope": "training split only; per site; finite values; population standard deviation; zero standard deviation replaced with 1",
        "missing_value_policy": "non-finite values are replaced with the fitted training mean at transform time; raw stale index values remain missing",
        "site_checks": stats_checks,
        "canonical_dataset_modified": False,
        "legacy_normalizer_modified": False,
        "model_training_or_forward_pass": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.config), indent=2))


if __name__ == "__main__":
    main()
