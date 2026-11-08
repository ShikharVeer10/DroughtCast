"""Paired clustered bootstrap utilities for independent forecast comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapConfig:
    """Predeclared resampling settings."""

    block_columns: tuple[str, ...] = ("site", "target_month")
    iterations: int = 2000
    seed: int = 202610
    alpha: float = 0.05


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, preserving NaN inputs."""
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    order = np.argsort(values[valid])
    sorted_values = values[valid][order]
    adjusted = np.maximum.accumulate((len(sorted_values) - np.arange(len(sorted_values))) * sorted_values)
    adjusted = np.minimum(adjusted, 1.0)
    positions = np.flatnonzero(valid)[order]
    result[positions] = adjusted
    return result


def paired_cluster_bootstrap(
    frame,
    model_a: str,
    model_b: str,
    *,
    config: BootstrapConfig | None = None,
) -> dict[str, float | int | str]:
    """Estimate paired metric differences by resampling whole configured blocks."""
    config = config or BootstrapConfig()
    required = set(config.block_columns) | {"model", "prediction", "observed"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Comparison frame missing columns: {sorted(missing)}")
    pair = frame[frame["model"].isin((model_a, model_b))].copy()
    keys = list(config.block_columns)
    wide = pair.pivot_table(index=keys, columns="model", values=["prediction", "observed"], aggfunc="first")
    if (model_a not in wide["prediction"] or model_b not in wide["prediction"]):
        raise ValueError("Models do not have paired prediction blocks")
    wide = wide.dropna(subset=[("prediction", model_a), ("prediction", model_b), ("observed", model_a)])
    if len(wide) < 2:
        return {"model_a": model_a, "model_b": model_b, "n_blocks": len(wide),
                "estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "warning": "insufficient independent blocks"}
    observed = wide[("observed", model_a)].to_numpy(float)
    a = wide[("prediction", model_a)].to_numpy(float)
    b = wide[("prediction", model_b)].to_numpy(float)
    differences = np.mean(np.abs(a - observed), axis=0) - np.mean(np.abs(b - observed), axis=0)
    rng = np.random.default_rng(config.seed)
    samples = np.empty(config.iterations)
    for i in range(config.iterations):
        indices = rng.integers(0, len(wide), len(wide))
        samples[i] = np.mean(np.abs(a[indices] - observed[indices])) - np.mean(
            np.abs(b[indices] - observed[indices])
        )
    low, high = np.quantile(samples, [config.alpha / 2, 1 - config.alpha / 2])
    return {"model_a": model_a, "model_b": model_b, "n_blocks": len(wide),
            "estimate": float(differences), "ci_low": float(low), "ci_high": float(high),
            "warning": ""}
