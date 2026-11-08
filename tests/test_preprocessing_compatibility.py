import json

import numpy as np
import pandas as pd
import pytest
import yaml

from data.loader import load_site_table
from data.preprocessing import (
    NORMALIZER_METADATA_SCHEMA_VERSION,
    ClimateNormalizer,
    validate_configured_model_features,
    validate_normalizer_metadata,
)
from data.protocol import DATASET_PROTOCOL_VERSION, MODEL_FEATURES, SPLIT_BOUNDARIES
from inference import _load_recent_input


def test_current_config_and_protocol_share_one_exact_15_feature_order():
    with open("configs/config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    features = config["data"]["features"]
    validate_configured_model_features(features, "test config")
    assert len(features) == 15
    assert len(set(features)) == 15
    assert config["model"]["input_dim"] == 15


@pytest.mark.parametrize(
    "features,fragment",
    [
        (MODEL_FEATURES[:-1], "missing"),
        ([*MODEL_FEATURES, "surprise"], "unexpected"),
        (MODEL_FEATURES[::-1], "position_mismatches"),
        ([*MODEL_FEATURES, MODEL_FEATURES[0]], "duplicate_actual"),
    ],
)
def test_invalid_feature_contracts_fail_with_named_diagnostics(features, fragment):
    with pytest.raises(ValueError, match=fragment):
        validate_configured_model_features(features, "test config")


def test_normalizer_uses_training_statistics_and_imputes_missing_with_train_mean():
    train = np.array([[1.0, 10.0], [3.0, 20.0], [np.nan, 30.0]])
    normalizer = ClimateNormalizer().fit(train)
    validation_test = np.array([[1e9, -1e9]])
    fitted_after_extremes = ClimateNormalizer().fit(
        np.vstack([train[:2], validation_test])
    )
    np.testing.assert_allclose(normalizer.mean, [2.0, 20.0])
    assert not np.allclose(normalizer.mean, fitted_after_extremes.mean)
    normalized = normalizer.transform(np.array([[np.nan, 25.0], [np.inf, np.nan]]))
    assert normalized.dtype == np.float64
    np.testing.assert_allclose(normalized, [[0.0, 5.0 / np.sqrt(200.0 / 3.0)], [0.0, 0.0]])


def test_zero_standard_deviation_uses_unit_scale():
    normalizer = ClimateNormalizer().fit(np.array([[4.0, 1.0], [4.0, 3.0]]))
    np.testing.assert_array_equal(normalizer.std, [1.0, 1.0])
    np.testing.assert_allclose(normalizer.transform(np.array([[5.0, 2.0]])), [[1.0, 0.0]])


def test_normalizer_serialization_preserves_order_and_rejects_reorder():
    names = ["a", "b"]
    fitted = ClimateNormalizer().fit(np.array([[1.0, 10.0], [3.0, 20.0]]))
    encoded = fitted.to_dict(names)
    restored = ClimateNormalizer.from_dict(encoded, expected_features=names)
    np.testing.assert_array_equal(restored.mean, fitted.mean)
    np.testing.assert_array_equal(restored.std, fitted.std)
    with pytest.raises(ValueError, match="position_mismatches"):
        ClimateNormalizer.from_dict(encoded, expected_features=names[::-1])
    with pytest.raises(ValueError, match="Feature-name count"):
        fitted.to_dict(["only-one"])


def test_training_metadata_is_fitted_only_on_training_rows(tmp_path):
    dates = pd.date_range("1990-01-01", "2015-12-01", freq="MS")
    values = np.arange(len(dates), dtype=float)
    frame = pd.DataFrame({"date": dates, "site": "aurangabad"})
    for index, feature in enumerate(MODEL_FEATURES):
        frame[feature] = values + index
    post_train = frame["date"] > pd.Timestamp(SPLIT_BOUNDARIES["train_end"])
    frame.loc[post_train, MODEL_FEATURES] = 1e8
    for target in ("spei_3", "spei_6", "spei_12"):
        frame[target] = 0.0
    dataset = tmp_path / "synthetic_monthly.csv"
    metadata_path = tmp_path / "normalizers.json"
    frame.to_csv(dataset, index=False)

    load_site_table(
        dataset,
        features=MODEL_FEATURES,
        seq_len=6,
        forecast_steps=4,
        metadata_path=metadata_path,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stats = metadata["normalizers"]["aurangabad"]
    train_values = frame.loc[~post_train, MODEL_FEATURES].to_numpy(dtype=float)
    np.testing.assert_allclose(stats["mean"], train_values.mean(axis=0))
    np.testing.assert_allclose(stats["std"], train_values.std(axis=0))
    assert metadata["normalizer_schema_version"] == NORMALIZER_METADATA_SCHEMA_VERSION
    assert metadata["dataset_protocol_version"] == DATASET_PROTOCOL_VERSION
    assert metadata["training_dates_by_site"]["aurangabad"]["row_count"] == 252
    assert metadata["training_dates_by_site"]["aurangabad"]["end"] == "2010-12"
    assert metadata["training_dataset_sha256"]


def test_normalizer_provenance_validation_rejects_dataset_or_protocol_mismatch():
    stats = ClimateNormalizer().fit(np.arange(30, dtype=float).reshape(2, 15)).to_dict(
        MODEL_FEATURES
    )
    metadata = {
        "normalizer_schema_version": NORMALIZER_METADATA_SCHEMA_VERSION,
        "dataset_protocol_version": DATASET_PROTOCOL_VERSION,
        "training_dataset_sha256": "abc",
        "features": list(MODEL_FEATURES),
        "split_boundaries": SPLIT_BOUNDARIES,
        "training_split": "train",
        "training_dates_by_site": {"aurangabad": {"row_count": 252, "start": "1990-01", "end": "2010-12"}},
        "seq_len": 6,
        "forecast_steps": 4,
        "normalizers": {"aurangabad": stats},
    }
    validate_normalizer_metadata(
        metadata,
        MODEL_FEATURES,
        expected_dataset_sha256="abc",
        expected_split_boundaries=SPLIT_BOUNDARIES,
        expected_seq_len=6,
        expected_forecast_steps=4,
    )
    with pytest.raises(ValueError, match="dataset hash"):
        validate_normalizer_metadata(metadata, MODEL_FEATURES, expected_dataset_sha256="wrong")
    with pytest.raises(ValueError, match="protocol version"):
        invalid = {**metadata, "dataset_protocol_version": DATASET_PROTOCOL_VERSION - 1}
        validate_normalizer_metadata(invalid, MODEL_FEATURES)


def test_stale_index_remains_missing_before_train_mean_imputation():
    normalizer = ClimateNormalizer().fit(np.array([[0.2], [0.4], [0.6]]))
    stale_index = np.array([[np.nan]])
    transformed = normalizer.transform(stale_index)
    assert np.isnan(stale_index[0, 0])
    assert transformed[0, 0] == 0.0


def test_inference_uses_training_sentinel_policy_and_feature_order(tmp_path):
    dates = pd.date_range("2026-05-01", periods=6, freq="MS")
    base = np.arange(1, len(MODEL_FEATURES) + 1, dtype=float)
    frame = pd.DataFrame({"date": dates, "site": "aurangabad"})
    for feature, value in zip(MODEL_FEATURES, base, strict=True):
        frame[feature] = value
    frame.loc[0, MODEL_FEATURES[0]] = -999.0
    frame.loc[5, ["dmi", "pdo"]] = np.nan
    input_path = tmp_path / "input.csv"
    frame.to_csv(input_path, index=False)

    fitted = ClimateNormalizer().fit(np.stack([base - 1.0, base + 1.0]))
    normalizer_path = tmp_path / "normalizers.json"
    metadata = {
        "normalizer_schema_version": NORMALIZER_METADATA_SCHEMA_VERSION,
        "dataset_protocol_version": DATASET_PROTOCOL_VERSION,
        "features": list(MODEL_FEATURES),
        "split_boundaries": SPLIT_BOUNDARIES,
        "training_split": "train",
        "training_dates_by_site": {
            "aurangabad": {"row_count": 252, "start": "1990-01", "end": "2010-12"}
        },
        "seq_len": 6,
        "forecast_steps": 4,
        "normalizers": {"aurangabad": fitted.to_dict(MODEL_FEATURES)},
    }
    normalizer_path.write_text(json.dumps(metadata), encoding="utf-8")

    values, actual_dates, _ = _load_recent_input(
        input_path, "aurangabad", MODEL_FEATURES, 6, normalizer_path
    )
    assert actual_dates == list(dates)
    assert values.shape == (6, 15)
    assert values.dtype == np.float32
    assert np.isfinite(values).all()
    assert values[0, 0] == 0.0  # -999 is treated as missing, then train-mean imputed.
    assert values[-1, MODEL_FEATURES.index("dmi")] == 0.0
    assert values[-1, MODEL_FEATURES.index("pdo")] == 0.0
