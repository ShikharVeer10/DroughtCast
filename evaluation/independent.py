"""Read-only preparation and validation for the frozen independent evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from data.preprocessing import validate_feature_order, validate_normalizer_metadata
from utils.artifacts import load_checkpoint, sha256_file, validate_checkpoint_metadata

INDEPENDENT_ORIGINS = ("2026-06", "2026-07", "2026-08", "2026-09", "2026-10")
SCALES = ("spei_3", "spei_6", "spei_12")
OUTPUT_COLUMNS = (
    "origin",
    "site",
    "scale",
    "lead",
    "observed",
    "predicted",
    "model",
    "rmse",
    "mae",
    "skill_vs_persistence",
    "precision",
    "recall",
    "f1",
    "severe_event_recall",
    "prediction_variance",
    "target_variance",
    "variance_ratio",
    "interval_coverage",
    "interval_width",
)


@dataclass(frozen=True)
class IndependentEvaluationSpec:
    """Immutable protocol contract for the five-origin evaluation."""

    origins: tuple[str, ...] = INDEPENDENT_ORIGINS
    seq_len: int = 6
    forecast_steps: int = 4
    sites: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    scales: tuple[str, ...] = SCALES

    def validate(self, consumed_test_end: str = "2025-12") -> None:
        if self.origins != INDEPENDENT_ORIGINS:
            raise ValueError(f"origins must equal {INDEPENDENT_ORIGINS}")
        if self.seq_len != 6 or self.forecast_steps != 4:
            raise ValueError("Independent protocol requires six inputs and four leads")
        if len(set(self.sites)) != 12:
            raise ValueError("Independent protocol requires 12 unique sites")
        if len(self.features) != 15:
            raise ValueError("Independent protocol requires 15 ordered features")
        if self.scales != SCALES:
            raise ValueError(f"scales must equal {SCALES}")
        if any(origin <= consumed_test_end for origin in self.origins):
            raise ValueError("Independent origins overlap the consumed test period")

    def windows(self) -> list[dict[str, str]]:
        self.validate()
        result = []
        for origin in self.origins:
            year, month = map(int, origin.split("-"))
            start = (year, month - self.seq_len + 1)
            if start[1] <= 0:
                start = (year - 1, start[1] + 12)
            end_month = month + self.forecast_steps
            end_year = year + (end_month - 1) // 12
            end_month = (end_month - 1) % 12 + 1
            result.append(
                {
                    "origin": origin,
                    "input_start": f"{start[0]:04d}-{start[1]:02d}",
                    "input_end": origin,
                    "target_start": f"{year + (month // 12):04d}-{month % 12 + 1:02d}",
                    "target_end": f"{end_year:04d}-{end_month:02d}",
                }
            )
        return result


def validate_candidate_windows(
    rows: list[dict[str, Any]], spec: IndependentEvaluationSpec
) -> None:
    """Validate metadata rows without reading targets or running inference."""
    spec.validate()
    expected = {item["origin"]: item for item in spec.windows()}
    seen = set()
    for row in rows:
        origin = row.get("origin")
        if origin not in expected or origin in seen:
            raise ValueError(f"Invalid or duplicate independent origin: {origin}")
        seen.add(origin)
        if row.get("input_start") != expected[origin]["input_start"]:
            raise ValueError(f"Invalid input window for {origin}")
        if row.get("input_end") != origin:
            raise ValueError(f"Input end must equal origin for {origin}")
        if row.get("target_end") != expected[origin]["target_end"]:
            raise ValueError(f"Invalid target window for {origin}")
    if seen != set(expected):
        raise ValueError("All five independent origins are required")


def validate_output_schema(rows: list[dict[str, Any]]) -> None:
    """Reject missing schema fields while preserving unavailable metrics as NaN."""
    missing = set(OUTPUT_COLUMNS).difference(rows[0] if rows else OUTPUT_COLUMNS)
    if missing:
        raise ValueError(f"Independent output schema missing columns: {sorted(missing)}")
    for row in rows:
        if row["lead"] not in {"T+1", "T+2", "T+3", "T+4"}:
            raise ValueError(f"Invalid forecast lead: {row['lead']}")
        if row["scale"] not in SCALES:
            raise ValueError(f"Invalid scale: {row['scale']}")


def classification_metrics(predicted: np.ndarray, observed: np.ndarray, threshold: float = -1.0) -> dict[str, float]:
    """Return explicit precision, recall and F1; undefined values are NaN."""
    p = np.asarray(predicted, dtype=float)
    y = np.asarray(observed, dtype=float)
    valid = np.isfinite(p) & np.isfinite(y)
    p, y = p[valid] <= threshold, y[valid] <= threshold
    tp, fp, fn = np.sum(p & y), np.sum(p & ~y), np.sum(~p & y)
    precision = float(tp / (tp + fp)) if tp + fp else float("nan")
    recall = float(tp / (tp + fn)) if tp + fn else float("nan")
    f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1}


def load_classical_artifact(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_features: list[str] | None = None,
) -> Any:
    """Load one serialized estimator and verify its file hash and feature contract."""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Classical artifact does not exist: {artifact}")
    digest = sha256_file(artifact)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Classical artifact hash mismatch: {artifact}")
    envelope = joblib.load(artifact)
    if not isinstance(envelope, dict) or "estimator" not in envelope:
        raise ValueError(f"Classical artifact has no estimator envelope: {artifact}")
    estimator = envelope["estimator"]
    metadata = envelope.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError(f"Classical artifact metadata is invalid: {artifact}")
    expected_method = artifact.name.split("_")[0]
    if artifact.name.startswith("random_forest"):
        expected_method = "random_forest"
    if metadata.get("method") != expected_method:
        raise ValueError(f"Classical artifact method mismatch: {artifact}")
    if expected_features is not None:
        saved = getattr(estimator, "feature_names_in_", None)
        if saved is not None:
            validate_feature_order(saved.tolist(), expected_features, str(artifact))
        elif getattr(estimator, "n_features_in_", None) not in (
            None,
            len(expected_features) * 6,
        ):
            raise ValueError(f"Classical artifact feature count mismatch: {artifact}")
        elif metadata.get("feature_variant") == "all_features" and len(expected_features) != 15:
            raise ValueError(f"All-feature artifact has an incompatible feature contract: {artifact}")
        elif metadata.get("feature_variant") == "without_climate_indices" and len(expected_features) != 11:
            raise ValueError(f"Reduced-feature artifact has an incompatible feature contract: {artifact}")
    return estimator


def validate_neural_checkpoint(
    path: str | Path,
    config: dict,
    *,
    dataset_path: str | Path,
    expected_sha256: str | None = None,
) -> dict:
    """Validate checkpoint metadata/state compatibility without constructing a forward pass."""
    checkpoint = Path(path)
    if expected_sha256 is not None and sha256_file(checkpoint) != expected_sha256:
        raise ValueError(f"Neural checkpoint hash mismatch: {checkpoint}")
    state = load_checkpoint(checkpoint, map_location="cpu")
    if "artifact_metadata" in state:
        validate_checkpoint_metadata(state, config, dataset_path)
    else:
        required = {
            "encoder.lstm1.weight_ih_l0": (256, len(config["data"]["features"])),
            "encoder.lstm2.weight_ih_l0": (128, 64),
            "decoders.0.2.weight": (
                config["data"]["forecast_steps"] * len(config["model"]["quantiles"]),
                64,
            ),
        }
        for key, shape in required.items():
            if key not in state or tuple(state[key].shape) != shape:
                raise ValueError(f"Neural checkpoint tensor contract mismatch: {checkpoint} ({key})")
    return {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "architecture_fingerprint": state.get("artifact_metadata", {}).get(
            "architecture_fingerprint", "tensor-contract-only"
        ),
        "features": state.get("artifact_metadata", {}).get(
            "features", list(config["data"]["features"])
        ),
    }


def validate_normalizer_file(
    path: str | Path,
    *,
    features: list[str],
    dataset_sha256: str,
    split_boundaries: dict[str, str],
    seq_len: int = 6,
    forecast_steps: int = 4,
) -> None:
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_normalizer_metadata(
        metadata,
        features,
        expected_dataset_sha256=dataset_sha256,
        expected_split_boundaries=split_boundaries,
        expected_seq_len=seq_len,
        expected_forecast_steps=forecast_steps,
    )


def _finite_pair_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    p = np.asarray(predicted, dtype=float)
    y = np.asarray(observed, dtype=float)
    valid = np.isfinite(p) & np.isfinite(y)
    p, y = p[valid], y[valid]
    if len(y) == 0:
        return {key: float("nan") for key in ("rmse", "mae", "bias", "correlation")}
    correlation = float("nan")
    if len(y) >= 2 and np.std(p) > 0 and np.std(y) > 0:
        correlation = float(np.corrcoef(p, y)[0, 1])
    return {
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mae": float(np.mean(np.abs(p - y))),
        "bias": float(np.mean(p - y)),
        "correlation": correlation,
    }


def _drought_category(values: np.ndarray) -> np.ndarray:
    bins = np.array([-np.inf, -2.0, -1.5, -1.0, 1.0, np.inf])
    return np.digitize(np.asarray(values, dtype=float), bins[1:-1], right=True)


def _classification_summary(predicted: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    p, y = np.asarray(predicted, dtype=float), np.asarray(observed, dtype=float)
    valid = np.isfinite(p) & np.isfinite(y)
    p, y = _drought_category(p[valid]), _drought_category(y[valid])
    labels = tuple(range(5))
    matrix = [[int(np.sum((y == row) & (p == col))) for col in labels] for row in labels]
    per_category = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[row][label] for row in labels if row != label)
        fn = sum(matrix[label][col] for col in labels if col != label)
        per_category[str(label)] = {
            "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
            "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else float("nan"),
        }
    f1s = np.array([v["f1"] for v in per_category.values()], dtype=float)
    support = np.array([sum(row) for row in matrix], dtype=float)
    valid_f1 = np.isfinite(f1s)
    return {
        "accuracy": float(np.mean(p == y)) if len(y) else float("nan"),
        "macro_f1": float(np.mean(f1s[valid_f1])) if valid_f1.any() else float("nan"),
        "weighted_f1": float(np.average(f1s[valid_f1], weights=support[valid_f1]))
        if valid_f1.any() and support[valid_f1].sum() else float("nan"),
        "severe_recall": _binary_recall(p <= 1, y <= 1),
        "severe_or_extreme_recall": _binary_recall(p <= 2, y <= 2),
        "per_category": per_category,
        "confusion_matrix": matrix,
    }


def _binary_recall(predicted: np.ndarray, observed: np.ndarray) -> float:
    tp = int(np.sum(predicted & observed))
    fn = int(np.sum(~predicted & observed))
    return float(tp / (tp + fn)) if tp + fn else float("nan")


def _quantile_summary(frame: pd.DataFrame) -> dict[str, float]:
    values = frame.pivot_table(index=["site", "forecast_origin", "target_month", "scale", "lead"],
                                columns="quantile", values="prediction", aggfunc="first")
    required = {0.1, 0.5, 0.9}
    if not required.issubset(values.columns):
        return {"quantile_loss": float("nan"), "coverage": float("nan"),
                "interval_width": float("nan"), "crossing_rate": float("nan")}
    q10, q50, q90 = (values[q] for q in (0.1, 0.5, 0.9))
    crossing = ((q10 > q50) | (q50 > q90)).to_numpy()
    return {"quantile_loss": float("nan"), "coverage": float("nan"),
            "interval_width": float(np.nanmean(q90 - q10)),
            "crossing_rate": float(np.mean(crossing)) if len(crossing) else float("nan")}


def run_independent_evaluation(
    forecast_dir: str | Path,
    dataset_path: str | Path,
    freeze_manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Evaluate frozen forecasts only against an accepted, hash-bound dataset."""
    manifest = json.loads(Path(freeze_manifest_path).read_text(encoding="utf-8"))
    if manifest.get("status") != "accepted":
        raise ValueError("Independent dataset is not frozen with status=accepted")
    dataset = pd.read_csv(dataset_path)
    required = {"site", "target_month", "scale", "observed"}
    if not required.issubset(dataset.columns):
        raise ValueError(f"Dataset missing columns: {sorted(required - set(dataset.columns))}")
    if dataset.duplicated(["site", "target_month", "scale"]).any():
        raise ValueError("Duplicate observed site-month-scale keys")
    if dataset["observed"].isna().any():
        raise ValueError("Observed targets contain missing values")
    expected_hash = manifest.get("dataset_sha256")
    if expected_hash and sha256_file(dataset_path) != expected_hash:
        raise ValueError("Independent dataset hash mismatch")
    frames = []
    for name in ("classical_forecasts.csv", "neural_forecasts.csv"):
        frame = pd.read_csv(Path(forecast_dir) / name)
        frame["quantile"] = pd.to_numeric(frame.get("quantile"), errors="coerce")
        frames.append(frame)
    forecasts = pd.concat(frames, ignore_index=True)
    forecast_required = {"forecast_origin", "lead", "site", "target_month", "scale",
                         "prediction", "model"}
    if not forecast_required.issubset(forecasts.columns):
        raise ValueError(f"Forecasts missing columns: {sorted(forecast_required - set(forecasts.columns))}")
    forecast_origins = tuple(sorted(forecasts["forecast_origin"].dropna().unique()))
    if not set(forecast_origins).issubset(set(INDEPENDENT_ORIGINS)):
        raise ValueError("Forecast contains an origin outside the independent protocol")
    if not forecast_origins:
        raise ValueError("Forecast contains no origins")
    key = ["site", "target_month", "scale"]
    merged = forecasts.merge(dataset, on=key, how="left", validate="many_to_one")
    if merged["observed"].isna().any():
        raise ValueError("Forecasts have incomplete observed-target coverage")
    merged["metric_group"] = merged["model"].astype(str)
    groupings = {
        "overall": ["model"],
        "model_scale": ["model", "scale"],
        "model_lead": ["model", "lead"],
        "model_site": ["model", "site"],
        "model_origin": ["model", "forecast_origin"],
        "model_site_scale_lead": ["model", "site", "scale", "lead"],
        "model_origin_scale_lead": ["model", "forecast_origin", "scale", "lead"],
    }
    tables = {}
    for level, columns in groupings.items():
        rows = []
        for group_keys, group in merged.groupby(columns, sort=True, dropna=False):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            metrics = _finite_pair_metrics(group["prediction"], group["observed"])
            metrics.update(dict(zip(columns, group_keys)))
            metrics.update({"level": level, "n": len(group)})
            rows.append(metrics)
        tables[level] = pd.DataFrame(rows)
    result = tables["overall"]
    if output_dir is not None:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        for level, table in tables.items():
            table.to_csv(out / f"{level}_metrics.csv", index=False)
        classification_rows = []
        for keys, group in merged.groupby(["model", "scale"], sort=True):
            summary = _classification_summary(group["prediction"], group["observed"])
            classification_rows.append({
                "model": keys[0], "scale": keys[1], "n": len(group),
                "accuracy": summary["accuracy"], "macro_f1": summary["macro_f1"],
                "weighted_f1": summary["weighted_f1"],
                "severe_recall": summary["severe_recall"],
                "severe_or_extreme_recall": summary["severe_or_extreme_recall"],
                "confusion_matrix": json.dumps(summary["confusion_matrix"]),
            })
        pd.DataFrame(classification_rows).to_csv(out / "classification_metrics.csv", index=False)
        neural = merged[merged["model"].astype(str).str.contains("neural", case=False, na=False)]
        if not neural.empty:
            neural["quantile"] = pd.to_numeric(neural["quantile"], errors="coerce")
            qrows = []
            for keys, group in neural.groupby(["model", "scale"], sort=True):
                if not {0.1, 0.5, 0.9}.issubset(set(group["quantile"].dropna())):
                    continue
                q10, q50, q90 = (group[group["quantile"] == q]["prediction"].to_numpy()
                                 for q in (0.1, 0.5, 0.9))
                obs = group.drop_duplicates(["site", "forecast_origin", "target_month", "lead"])["observed"].to_numpy()
                qloss = np.nanmean(np.maximum(0.1 * (obs - q10), (0.1 - 1) * (obs - q10)))
                qrows.append({"model": keys[0], "scale": keys[1], "n": len(obs),
                              "pinball_loss_q10": float(qloss),
                              "coverage_q10_q90": float(np.mean((obs >= q10) & (obs <= q90))),
                              "mean_interval_width": float(np.mean(q90 - q10)),
                              "crossing_frequency": float(np.mean((q10 > q50) | (q50 > q90)))})
            pd.DataFrame(qrows).to_csv(out / "quantile_metrics.csv", index=False)
    return result
