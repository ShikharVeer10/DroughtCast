"""Descriptive robustness checks for frozen forecasts and future evaluations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_behavior_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize finite forecast behavior by model, site, scale, and lead."""
    required = {"model", "site", "scale", "lead", "prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Forecast frame missing columns: {sorted(missing)}")
    values = frame.copy()
    values["prediction"] = pd.to_numeric(values["prediction"], errors="coerce")
    values = values[np.isfinite(values["prediction"])]
    return (
        values.groupby(["model", "site", "scale", "lead"], as_index=False)["prediction"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )


def model_disagreement(frame: pd.DataFrame) -> pd.DataFrame:
    """Return spread across models for each forecast key."""
    required = {"model", "site", "scale", "lead", "prediction"}
    if not required.issubset(frame.columns):
        raise ValueError("Forecast frame lacks disagreement columns")
    return (
        frame.groupby(["site", "scale", "lead"], as_index=False)["prediction"]
        .agg(model_count="count", model_mean="mean", model_std="std",
             model_min="min", model_max="max")
        .assign(model_range=lambda x: x["model_max"] - x["model_min"])
    )


def stability_summary(frame: pd.DataFrame, value_column: str = "metric") -> dict[str, float | int]:
    """Summarize future evaluation variation without interpreting accuracy."""
    if value_column not in frame:
        raise ValueError(f"Missing stability value column: {value_column}")
    values = pd.to_numeric(frame[value_column], errors="coerce").dropna()
    return {
        "n": len(values),
        "mean": float(values.mean()) if len(values) else float("nan"),
        "std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
        "min": float(values.min()) if len(values) else float("nan"),
        "max": float(values.max()) if len(values) else float("nan"),
    }
