"""Validation-calibrated conformalized quantile regression helpers."""
from __future__ import annotations

import numpy as np


def calibrate_cqr(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray,
                  quantiles: list[float], coverage: float = 0.8) -> np.ndarray:
    """Return lead-wise CQR corrections from calibration (validation) data.

    prediction: [sample, lead, quantile], target/mask: [sample, lead].
    Uses the finite-sample higher order statistic (ceil((n+1)*coverage)).
    """
    if prediction.ndim != 3 or prediction.shape[-1] != len(quantiles):
        raise ValueError("prediction must have shape [sample, lead, len(quantiles)]")
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie strictly between 0 and 1")
    if not quantiles:
        raise ValueError("quantile list must be non-empty")
    lo_idx, hi_idx = quantiles.index(min(quantiles)), quantiles.index(max(quantiles))
    corrections = []
    for lead in range(target.shape[1]):
        valid = mask[:, lead] & np.isfinite(target[:, lead])
        if valid.sum() == 0:
            corrections.append(float("nan"))
            continue
        score = np.maximum(prediction[valid, lead, lo_idx] - target[valid, lead],
                           target[valid, lead] - prediction[valid, lead, hi_idx])
        rank = int(np.ceil((len(score) + 1) * coverage))
        # If the finite-sample conformal rank exceeds n, the conservative
        # correction is +infinity; clipping to n loses the stated guarantee.
        corrections.append(float("inf") if rank > len(score) else
                           float(np.partition(score, rank - 1)[rank - 1]))
    return np.asarray(corrections)


def apply_cqr(prediction: np.ndarray, corrections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = 0
    hi = prediction.shape[-1] - 1
    delta = corrections.reshape(1, -1)
    # CQR corrections can be negative when the base interval is over-wide.
    # Constrain it to zero for valid, nested prediction intervals.
    delta = np.maximum(delta, 0.0)
    return prediction[:, :, lo] - delta, prediction[:, :, hi] + delta


def interval_metrics(lower: np.ndarray, upper: np.ndarray, target: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(target)
    if not valid.any():
        return {"coverage": float("nan"), "width": float("nan"), "n": 0}
    return {"coverage": float(np.mean((target[valid] >= lower[valid]) & (target[valid] <= upper[valid]))),
            "width": float(np.mean(upper[valid] - lower[valid])), "n": int(valid.sum())}


def pinball_score(prediction: np.ndarray, target: np.ndarray, quantiles: list[float],
                  mask: np.ndarray | None = None) -> float:
    error = target[..., None] - prediction
    q = np.asarray(quantiles).reshape((1,) * target.ndim + (-1,))
    loss = np.maximum(q * error, (q - 1) * error)
    valid = np.isfinite(target) if mask is None else (mask & np.isfinite(target))
    return float(np.mean(loss[valid])) if valid.any() else float("nan")


def crps_from_quantiles(prediction: np.ndarray, target: np.ndarray, quantiles: list[float],
                        mask: np.ndarray | None = None) -> float:
    """Quantile approximation to CRPS: twice integrated pinball loss."""
    if len(quantiles) < 2:
        raise ValueError("at least two quantiles are required")
    error = target[..., None] - prediction
    q = np.asarray(quantiles).reshape((1,) * target.ndim + (-1,))
    losses = np.maximum(q * error, (q - 1) * error)
    valid = np.isfinite(target) if mask is None else (mask & np.isfinite(target))
    if not valid.any():
        return float("nan")
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    per_case = integrate(losses[valid], x=np.asarray(quantiles), axis=-1) * 2
    return float(np.mean(per_case))
