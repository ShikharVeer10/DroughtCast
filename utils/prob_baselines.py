"""Probabilistic baselines for monthly drought forecasts."""

from __future__ import annotations

import numpy as np


def persistence_residual_quantiles(
    last_observed: np.ndarray,
    train_last_observed: np.ndarray,
    train_targets: np.ndarray,
    train_mask: np.ndarray,
    quantiles: list[float],
) -> np.ndarray:
    """Predict persistence plus train-only, scale/lead residual quantiles.

    Shapes: last observed [N, scale], train targets/mask [N, scale, lead].
    Returns quantile forecasts [N, scale, lead, quantile].
    """
    last = np.asarray(last_observed, dtype=float)
    train_last = np.asarray(train_last_observed, dtype=float)
    targets = np.asarray(train_targets, dtype=float)
    mask = np.asarray(train_mask, dtype=bool)
    q = np.asarray(quantiles, dtype=float)
    if last.ndim != 2 or train_last.ndim != 2 or targets.ndim != 3 or mask.shape != targets.shape:
        raise ValueError("Expected [sample, scale] persistence values and [sample, scale, lead] train targets/masks")
    if train_last.shape != targets.shape[:2] or last.shape[1] != targets.shape[1]:
        raise ValueError("Persistence values and training target scales do not match")
    if q.ndim != 1 or not np.all(np.diff(q) > 0) or np.any((q <= 0) | (q >= 1)):
        raise ValueError("quantiles must be strictly increasing values in (0, 1)")
    residuals = targets - train_last[..., None]
    output = np.full((len(last), targets.shape[1], targets.shape[2], len(q)), np.nan)
    for scale in range(targets.shape[1]):
        for lead in range(targets.shape[2]):
            valid = mask[:, scale, lead] & np.isfinite(residuals[:, scale, lead])
            if valid.sum() < 10:
                raise ValueError(f"Too few training residuals for scale={scale}, lead={lead}: {valid.sum()}")
            offsets = np.quantile(residuals[valid, scale, lead], q, method="linear")
            output[:, scale, lead, :] = last[:, scale, None] + offsets[None, :]
    return output
