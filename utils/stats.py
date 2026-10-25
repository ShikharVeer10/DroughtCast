"""Paired uncertainty, Diebold-Mariano, and Wilcoxon Signed-Rank tests for forecast comparisons."""
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union


def wilcoxon_signed_rank_test(
    loss_model: Union[List[float], np.ndarray],
    loss_baseline: Union[List[float], np.ndarray],
    alternative: str = "less"
) -> Dict[str, Union[float, int, bool]]:
    """
    Non-parametric Wilcoxon Signed-Rank Test on paired forecast error losses.
    Testing H0: Median loss difference is zero vs H1: Model loss < Baseline loss ('less').
    Returns test statistic, exact/asymptotic p-value, and significance flag (p < 0.01).
    """
    m = np.asarray(loss_model, dtype=float)
    b = np.asarray(loss_baseline, dtype=float)
    diff = m - b
    valid = np.isfinite(diff) & (diff != 0.0)
    d = diff[valid]
    n = len(d)
    if n < 5:
        return {
            "n": n,
            "statistic": float("nan"),
            "p_value": float("nan"),
            "significant_01": False,
            "significant_05": False
        }
    res = stats.wilcoxon(d, alternative=alternative)
    return {
        "n": n,
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "significant_01": bool(res.pvalue < 0.01),
        "significant_05": bool(res.pvalue < 0.05),
        "mean_diff": float(np.mean(diff[np.isfinite(diff)]))
    }


def dm_hac(loss_model, loss_baseline, horizon: int = 1) -> dict:
    """Two-sided DM statistic with Newey-West HAC variance and Harvey correction."""
    d = np.asarray(loss_model, dtype=float) - np.asarray(loss_baseline, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return {"n": n, "statistic": float("nan"), "p_value": float("nan")}
    centered = d - d.mean()
    lag = max(0, min(horizon - 1, n - 1))
    long_run = float(np.dot(centered, centered) / n)
    for k in range(1, lag + 1):
        covariance = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - k / (lag + 1)
        long_run += 2 * weight * covariance
    if long_run <= 0:
        return {"n": n, "statistic": float("nan"), "p_value": float("nan")}
    statistic = d.mean() / np.sqrt(long_run / n)
    correction = np.sqrt(max((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n, 0))
    statistic *= correction
    # Two-sided normal approximation
    from math import erf, sqrt
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(statistic) / sqrt(2))))
    return {"n": n, "statistic": float(statistic), "p_value": float(p_value),
            "mean_loss_difference": float(d.mean())}


def paired_block_bootstrap_rmse_ci(prediction, baseline, target, groups,
                                   n_boot: int = 2000, seed: int = 123) -> dict:
    """Bootstrap paired RMSE difference by site-year blocks to reduce overlap leakage."""
    p, b, y = map(lambda x: np.asarray(x, dtype=float), (prediction, baseline, target))
    groups = np.asarray(groups)
    valid = np.isfinite(p) & np.isfinite(b) & np.isfinite(y)
    p, b, y, groups = p[valid], b[valid], y[valid], groups[valid]
    unique = np.unique(groups)
    if len(unique) < 2:
        return {"n_blocks": int(len(unique)), "rmse_difference_ci95": [float("nan"), float("nan")]}
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[g] for g in chosen])
        diffs[i] = np.sqrt(np.mean((p[rows] - y[rows]) ** 2)) - np.sqrt(np.mean((b[rows] - y[rows]) ** 2))
    return {"n_blocks": int(len(unique)), "rmse_difference_ci95": np.quantile(diffs, [.025, .975]).tolist()}
