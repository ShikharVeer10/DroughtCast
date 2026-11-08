"""Failure diagnostics for observed prediction/evaluation tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def residual_summary(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "site", "scale", "lead", "prediction", "observed"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Failure frame missing columns: {sorted(missing)}")
    data = frame.copy()
    data["prediction"] = pd.to_numeric(data["prediction"], errors="coerce")
    data["observed"] = pd.to_numeric(data["observed"], errors="coerce")
    data["residual"] = data["prediction"] - data["observed"]
    data = data[np.isfinite(data["residual"])]
    return data.groupby(["model", "site", "scale", "lead"], as_index=False).agg(
        n=("residual", "size"),
        bias=("residual", "mean"),
        mae=("residual", lambda x: float(np.mean(np.abs(x)))),
        rmse=("residual", lambda x: float(np.sqrt(np.mean(x**2)))),
        underprediction_count=("residual", lambda x: int(np.sum(x < 0))),
        overprediction_count=("residual", lambda x: int(np.sum(x > 0))),
    )


def drought_failure_summary(frame: pd.DataFrame, threshold: float = -1.5) -> dict[str, float | int]:
    required = {"prediction", "observed"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Failure frame missing columns: {sorted(required - set(frame.columns))}")
    p = pd.to_numeric(frame["prediction"], errors="coerce").to_numpy()
    y = pd.to_numeric(frame["observed"], errors="coerce").to_numpy()
    valid = np.isfinite(p) & np.isfinite(y)
    predicted, actual = p[valid] <= threshold, y[valid] <= threshold
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    return {
        "n": int(valid.sum()),
        "threshold": threshold,
        "hits": tp,
        "false_alarms": fp,
        "missed_events": fn,
        "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
    }


def require_observed_evaluation(frame: pd.DataFrame) -> None:
    """Prevent prospective claims when observed targets are absent."""
    if "observed" not in frame or pd.to_numeric(frame["observed"], errors="coerce").isna().all():
        raise ValueError("Failure analysis requires genuine observed targets")
