import hashlib
import json
from pathlib import Path

import numpy as np

from data.protocol import DATASET_PROTOCOL_VERSION, MODEL_FEATURES, SPLIT_BOUNDARIES

NORMALIZER_METADATA_SCHEMA_VERSION = 2
DEFAULT_NORMALIZER_METADATA_PATH = "data/normalizers/maharashtra_protocol10.json"
INVALID_VALUE_SENTINELS = (-999.0, -99.0, -9999.0, -32768.0)


def replace_invalid_sentinels(frame):
    return frame.replace(list(INVALID_VALUE_SENTINELS), np.nan)


def validate_feature_order(actual, expected, context: str = "normalizer") -> None:
    actual = list(actual or [])
    expected = list(expected or [])
    duplicates = sorted({name for name in actual if actual.count(name) > 1})
    expected_duplicates = sorted(
        {name for name in expected if expected.count(name) > 1}
    )
    missing = [name for name in expected if name not in actual]
    unexpected = [name for name in actual if name not in expected]
    positions = [
        {"position": index, "actual": actual[index], "expected": expected[index]}
        for index in range(min(len(actual), len(expected)))
        if actual[index] != expected[index]
    ]
    if duplicates or expected_duplicates or missing or unexpected or positions or len(actual) != len(expected):
        raise ValueError(
            f"{context} feature order mismatch: "
            f"actual={actual}; expected={expected}; "
            f"position_mismatches={positions}; missing={missing}; "
            f"unexpected={unexpected}; duplicate_actual={duplicates}; "
            f"duplicate_expected={expected_duplicates}"
        )


def validate_configured_model_features(features, context: str) -> None:
    validate_feature_order(features, MODEL_FEATURES, context)


def normalizer_provenance(
    dataset_path,
    features: list[str],
    training_dates_by_site: dict[str, dict[str, str | int]],
    seq_len: int,
    forecast_steps: int,
    split_boundaries: dict[str, str] | None = None,
) -> dict:
    """Build provenance for stats fitted by the loader on train-split rows."""
    path = Path(dataset_path)
    digest = None
    if path.is_file():
        hasher = hashlib.sha256()
        with open(path, "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
    descriptor = {
        "normalizer_schema_version": NORMALIZER_METADATA_SCHEMA_VERSION,
        "dataset_protocol_version": DATASET_PROTOCOL_VERSION,
        "training_dataset_sha256": digest,
        "training_dataset_path": str(path),
        "features": list(features),
        "split_boundaries": dict(split_boundaries or SPLIT_BOUNDARIES),
        "normalization": "per-site train-split nanmean and population nanstd; zero/invalid std replaced by 1.0",
        "imputation": "training_feature_mean",
        "training_split": "train",
        "training_dates_by_site": training_dates_by_site,
        "seq_len": int(seq_len),
        "forecast_steps": int(forecast_steps),
    }
    descriptor["preprocessing_fingerprint"] = hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return descriptor


def validate_normalizer_metadata(
    metadata: dict,
    expected_features: list[str],
    *,
    expected_dataset_sha256: str | None = None,
    expected_protocol_version: int = DATASET_PROTOCOL_VERSION,
    expected_split_boundaries: dict[str, str] | None = None,
    expected_seq_len: int | None = None,
    expected_forecast_steps: int | None = None,
) -> None:
    """Reject normalizer artifacts without compatible, auditable provenance."""
    if metadata.get("normalizer_schema_version") != NORMALIZER_METADATA_SCHEMA_VERSION:
        raise ValueError(
            "Normalizer metadata schema is missing or incompatible; "
            "generate a versioned artifact from training-split rows."
        )
    validate_feature_order(metadata.get("features", []), expected_features, "saved normalizer")
    if metadata.get("dataset_protocol_version") != expected_protocol_version:
        raise ValueError(
            "Normalizer dataset protocol version does not match the current protocol"
        )
    if expected_dataset_sha256 is not None and metadata.get("training_dataset_sha256") != expected_dataset_sha256:
        raise ValueError("Normalizer training dataset hash does not match the checkpoint dataset")
    if (
        expected_split_boundaries is not None
        and metadata.get("split_boundaries") != expected_split_boundaries
    ):
        raise ValueError("Normalizer split boundaries do not match the current protocol")
    if metadata.get("training_split") != "train":
        raise ValueError("Normalizer statistics are not declared as training-split only")
    training_end = metadata.get("split_boundaries", {}).get("train_end")
    for site, dates in metadata.get("training_dates_by_site", {}).items():
        if training_end is not None and dates.get("end", "9999-12") > str(training_end)[:7]:
            raise ValueError(
                f"Normalizer fit period for {site} extends beyond the training boundary"
            )
    if expected_seq_len is not None and metadata.get("seq_len") != expected_seq_len:
        raise ValueError("Normalizer sequence length does not match the current configuration")
    if (
        expected_forecast_steps is not None
        and metadata.get("forecast_steps") != expected_forecast_steps
    ):
        raise ValueError("Normalizer forecast steps do not match the current configuration")
    normalizers = metadata.get("normalizers")
    if not isinstance(normalizers, dict) or not normalizers:
        raise ValueError("Normalizer metadata has no per-site normalizer statistics")
    for stats in normalizers.values():
        ClimateNormalizer.from_dict(stats, expected_features=expected_features)


class ClimateNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("Expected [time, feature] data")
        finite = np.isfinite(data)
        counts = finite.sum(axis=0)
        if np.any(counts == 0):
            raise ValueError("Every feature needs at least one finite training observation")
        values = np.where(finite, data, np.nan)
        self.mean = np.nanmean(values, axis=0)
        self.std = np.nanstd(values, axis=0)
        self.std = np.where(np.isfinite(self.std) & (self.std > 0), self.std, 1.0)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ValueError("ClimateNormalizer must be fitted before transform")
        values = np.asarray(data, dtype=float)
        if values.shape[-1] != len(self.mean):
            raise ValueError("Feature count does not match fitted normalizer")
        imputed = np.where(np.isfinite(values), values, self.mean)
        return (imputed - self.mean) / self.std

    def to_dict(self, feature_names: list[str] | None = None):
        metadata = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "imputation": "training_feature_mean",
        }
        if feature_names is not None:
            if len(feature_names) != len(self.mean):
                raise ValueError("Feature-name count does not match normalizer statistics")
            metadata["features"] = list(feature_names)
        return metadata

    @classmethod
    def from_dict(cls, metadata: dict, expected_features: list[str] | None = None) -> "ClimateNormalizer":
        if expected_features is not None:
            validate_feature_order(
                metadata.get("features", []), expected_features, "saved normalizer"
            )
        normalizer = cls()
        normalizer.mean = np.asarray(metadata.get("mean"), dtype=float)
        normalizer.std = np.asarray(metadata.get("std"), dtype=float)
        if (
            normalizer.mean.ndim != 1
            or normalizer.std.shape != normalizer.mean.shape
            or not np.isfinite(normalizer.mean).all()
            or not np.isfinite(normalizer.std).all()
            or np.any(normalizer.std <= 0)
        ):
            raise ValueError("Invalid saved ClimateNormalizer statistics")
        return normalizer

