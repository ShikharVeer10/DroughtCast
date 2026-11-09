"""Development-stage drought category metrics for frozen prediction outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLDS = (-2.0, -1.5, -1.0, 1.0)
CATEGORIES = ("extreme", "severe", "moderate", "non_drought", "wet")


def categories(values: np.ndarray) -> np.ndarray:
    """Apply the repository's existing right-closed SPEI category boundaries."""
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, -1, dtype=np.int8)
    valid = np.isfinite(values)
    result[valid] = np.digitize(values[valid], THRESHOLDS, right=True)
    return result


def classification_metrics(frame: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    observed = categories(frame["observed"].to_numpy())
    predicted = categories(frame["prediction"].to_numpy())
    matrix = np.zeros((5, 5), dtype=int)
    valid = (observed >= 0) & (predicted >= 0)
    for actual, prediction in zip(observed[valid], predicted[valid]):
        matrix[actual, prediction] += 1
    per_category = []
    for index, name in enumerate(CATEGORIES):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision + recall) and precision + recall else np.nan
        per_category.append({"category": name, "support": int(matrix[index].sum()),
                             "precision": float(precision), "recall": float(recall), "f1": float(f1)})
    f1 = np.asarray([row["f1"] for row in per_category])
    support = matrix.sum(axis=1)
    valid_f1 = np.isfinite(f1)
    return matrix, [{
        "accuracy": float(np.mean(predicted[valid] == observed[valid])) if valid.any() else np.nan,
        "macro_f1": float(np.mean(f1[valid_f1])) if valid_f1.any() else np.nan,
        "weighted_f1": float(np.average(f1[valid_f1], weights=support[valid_f1])) if valid_f1.any() and support[valid_f1].sum() else np.nan,
        "severe_precision": per_category[1]["precision"],
        "severe_recall": per_category[1]["recall"],
        "false_alarm_rate": float((predicted[valid] == 1).sum() - matrix[1, 1]) / max(float((predicted[valid] != 1).sum()), 1.0),
        "missed_drought_rate": float(matrix[1:, 0].sum() + matrix[1:, 3:].sum()) / max(float(matrix[1:, :].sum()), 1.0),
        "per_category": per_category,
    }]
