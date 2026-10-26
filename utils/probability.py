"""Event-probability calibration and diagnostics."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def quantile_risk_below_threshold(quantiles: np.ndarray, threshold: float) -> np.ndarray:
    """Estimate lower-tail event risk by piecewise-linear interpolation of q10/q50/q90."""
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


def fit_isotonic_calibrator(raw_probability: np.ndarray,
                            event: np.ndarray) -> IsotonicRegression:
    raw = np.asarray(raw_probability, dtype=float)
    y = np.asarray(event, dtype=int)
    valid = np.isfinite(raw) & np.isfinite(y)
    if valid.sum() < 2 or np.unique(y[valid]).size < 2:
        raise ValueError("Probability calibration requires finite validation samples from both event classes")
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(np.clip(raw[valid], 0.0, 1.0), y[valid])
    return calibrator


def brier_score(probability: np.ndarray, event: np.ndarray) -> float:
    p, y = np.asarray(probability, dtype=float), np.asarray(event, dtype=float)
    valid = np.isfinite(p) & np.isfinite(y)
    if not valid.any():
        return float("nan")
    return float(np.mean((p[valid] - y[valid]) ** 2))


def reliability_table(probability: np.ndarray, event: np.ndarray,
                      bins: int = 10) -> list[dict[str, float | int]]:
    p, y = np.asarray(probability, dtype=float), np.asarray(event, dtype=float)
    valid = np.isfinite(p) & np.isfinite(y)
    p, y = np.clip(p[valid], 0.0, 1.0), y[valid]
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        selected = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if selected.any():
            rows.append({
                "bin": i, "count": int(selected.sum()),
                "mean_predicted": float(np.mean(p[selected])),
                "observed_frequency": float(np.mean(y[selected])),
            })
    return rows


def expected_calibration_error(probability: np.ndarray, event: np.ndarray,
                               bins: int = 10) -> float:
    table = reliability_table(probability, event, bins)
    total = sum(row["count"] for row in table)
    return float(sum(row["count"] * abs(row["mean_predicted"] - row["observed_frequency"])
                     for row in table) / total) if total else float("nan")


def calibration_slope_intercept(probability: np.ndarray, event: np.ndarray) -> dict[str, float]:
    p, y = np.asarray(probability, dtype=float), np.asarray(event, dtype=int)
    valid = np.isfinite(p) & np.isfinite(y)
    p = np.clip(p[valid], 1e-6, 1 - 1e-6)
    y = y[valid]
    if len(p) < 2 or np.unique(y).size < 2:
        return {"intercept": float("nan"), "slope": float("nan")}
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs").fit(logits, y)
    return {"intercept": float(model.intercept_[0]), "slope": float(model.coef_[0, 0])}
