"""Event-probability calibration and diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def quantile_risk_below_threshold(
    quantiles: np.ndarray, threshold: float
) -> np.ndarray:
    """LEGACY: interpolate lower-tail risk from q10/q50/q90."""
    values = np.asarray(quantiles, dtype=float)
    if values.shape[-1] != 3:
        raise ValueError("Expected exactly three quantiles ordered as q10, q50, q90")
    q10, q50, q90 = (values[..., i] for i in range(3))
    if not np.isfinite(values).all() or np.any(q10 > q50) or np.any(q50 > q90):
        raise ValueError("Quantile values must be finite and monotonically ordered")
    risk = np.empty_like(q50, dtype=float)
    risk[threshold <= q10] = 0.1
    risk[threshold >= q90] = 0.9
    low = (threshold > q10) & (threshold <= q50)
    high = (threshold > q50) & (threshold < q90)
    risk[low] = 0.1 + 0.4 * (threshold - q10[low]) / (q50[low] - q10[low] + 1e-8)
    risk[high] = 0.5 + 0.4 * (threshold - q50[high]) / (q90[high] - q50[high] + 1e-8)
    return np.clip(risk, 0.0, 1.0)


def event_z(quantiles: np.ndarray, threshold: float, min_scale: float = 0.05) -> np.ndarray:
    """Convert a threshold and q10/q50/q90 forecast to a signed standardized score."""
    values = np.asarray(quantiles, dtype=float)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError("Expected exactly three quantiles ordered as q10, q50, q90")
    q10, q50, q90 = (values[..., i] for i in range(3))
    if not np.isfinite(values).all() or np.any(q10 > q50) or np.any(q50 > q90):
        raise ValueError("Quantile values must be finite and monotonically ordered")
    scale_low = np.maximum((q50 - q10) / 1.2815515655, min_scale)
    scale_up = np.maximum((q90 - q50) / 1.2815515655, min_scale)
    return (threshold - q50) / np.where(threshold <= q50, scale_low, scale_up)


def raw_event_probability(z: np.ndarray) -> np.ndarray:
    return norm.cdf(z)


from dataclasses import dataclass
import warnings


@dataclass(frozen=True)
class PlattCalibrator:
    intercept: float
    slope: float
    n_samples: int
    n_events: int
    floor: float = 0.005
    ceiling: float = 0.995

    def predict(self, z: np.ndarray) -> np.ndarray:
        return np.clip(expit(self.intercept + self.slope * np.asarray(z, dtype=float)), self.floor, self.ceiling)

    def predict_unclipped(self, z: np.ndarray) -> np.ndarray:
        """Return the fitted Platt sigmoid before presentation-bound clipping."""
        return expit(self.intercept + self.slope * np.asarray(z, dtype=float))


def fit_platt_calibrator(z: np.ndarray, event: np.ndarray, min_events: int = 20) -> PlattCalibrator:
    scores, labels = np.asarray(z, dtype=float).reshape(-1), np.asarray(event, dtype=int).reshape(-1)
    valid = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[valid], labels[valid]
    if len(scores) < 2 or np.unique(labels).size < 2:
        raise ValueError("Platt calibration requires at least two samples from both event classes")
    model = LogisticRegression(C=1.0, solver="lbfgs").fit(scores.reshape(-1, 1), labels)
    events = int(labels.sum())
    slope = float(model.coef_[0, 0])
    if events < min_events:
        warnings.warn(f"Only {events} events; fewer than min_events={min_events}", UserWarning)
    if slope <= 0:
        warnings.warn("Fitted Platt calibration slope is nonpositive", UserWarning)
    return PlattCalibrator(float(model.intercept_[0]), slope, len(scores), events)


def fit_isotonic_calibrator(
    raw_probability: np.ndarray, event: np.ndarray
) -> IsotonicRegression:
    raw = np.asarray(raw_probability, dtype=float)
    y = np.asarray(event, dtype=int)
    valid = np.isfinite(raw) & np.isfinite(y)
    if valid.sum() < 2 or np.unique(y[valid]).size < 2:
        raise ValueError(
            "Probability calibration requires finite validation samples from both event classes"
        )
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(np.clip(raw[valid], 0.0, 1.0), y[valid])
    return calibrator


def brier_score(probability: np.ndarray, event: np.ndarray) -> float:
    p, y = np.asarray(probability, dtype=float), np.asarray(event, dtype=float)
    valid = np.isfinite(p) & np.isfinite(y)
    if not valid.any():
        return float("nan")
    return float(np.mean((p[valid] - y[valid]) ** 2))


def reliability_table(
    probability: np.ndarray, event: np.ndarray, bins: int = 10
) -> list[dict[str, float | int]]:
    p, y = np.asarray(probability, dtype=float), np.asarray(event, dtype=float)
    valid = np.isfinite(p) & np.isfinite(y)
    p, y = np.clip(p[valid], 0.0, 1.0), y[valid]
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        selected = (p >= edges[i]) & (
            (p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1])
        )
        if selected.any():
            rows.append(
                {
                    "bin": i,
                    "count": int(selected.sum()),
                    "mean_predicted": float(np.mean(p[selected])),
                    "observed_frequency": float(np.mean(y[selected])),
                }
            )
    return rows


def expected_calibration_error(
    probability: np.ndarray, event: np.ndarray, bins: int = 10
) -> float:
    table = reliability_table(probability, event, bins)
    total = sum(row["count"] for row in table)
    return (
        float(
            sum(
                row["count"] * abs(row["mean_predicted"] - row["observed_frequency"])
                for row in table
            )
            / total
        )
        if total
        else float("nan")
    )


def calibration_slope_intercept(
    probability: np.ndarray, event: np.ndarray
) -> dict[str, float]:
    p, y = np.asarray(probability, dtype=float), np.asarray(event, dtype=int)
    valid = np.isfinite(p) & np.isfinite(y)
    p = np.clip(p[valid], 1e-6, 1 - 1e-6)
    y = y[valid]
    if len(p) < 2 or np.unique(y).size < 2:
        return {"intercept": float("nan"), "slope": float("nan")}
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs").fit(logits, y)
    return {"intercept": float(model.intercept_[0]), "slope": float(model.coef_[0, 0])}
