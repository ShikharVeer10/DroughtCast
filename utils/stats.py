"""Paired uncertainty, Diebold-Mariano, and Wilcoxon Signed-Rank tests for forecast comparisons."""

import numpy as np
from scipy import stats


def effective_sample_size(values) -> int:
    """Estimate effective n using positive-lag autocorrelation contributions."""
    series = np.asarray(values, dtype=float).reshape(-1)
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 3 or np.std(series) == 0:
        return n
    centered = series - series.mean()
    variance = np.dot(centered, centered)
    total = 1.0
    for lag in range(1, n):
        rho = float(np.dot(centered[lag:], centered[:-lag]) / variance)
        if rho <= 0:
            break
        total += 2.0 * rho
    return max(1, min(n, int(round(n / total))))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment, preserving input order and NaN values."""
    indexed = [(i, float(p)) for i, p in enumerate(p_values) if np.isfinite(p)]
    indexed.sort(key=lambda item: item[1])
    adjusted = [float("nan")] * len(p_values)
    running = 0.0
    total = len(indexed)
    for rank, (index, p_value) in enumerate(indexed):
        running = max(running, min(1.0, (total - rank) * p_value))
        adjusted[index] = running
    return adjusted


def wilcoxon_signed_rank_test(
    loss_model: list[float] | np.ndarray,
    loss_baseline: list[float] | np.ndarray,
    alternative: str = "less",
) -> dict[str, float | int | bool]:
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
            "significant_05": False,
        }
    res = stats.wilcoxon(d, alternative=alternative)
    return {
        "n": n,
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "significant_01": bool(res.pvalue < 0.01),
        "significant_05": bool(res.pvalue < 0.05),
        "mean_diff": float(np.mean(diff[np.isfinite(diff)])),
    }


def dm_hac(
    loss_model, loss_baseline, horizon: int = 1, accumulation_scale: int = 1
) -> dict:
    """Two-sided Diebold-Mariano test for paired loss series.

    ``horizon`` controls the Newey-West truncation lag (horizon - 1). Pairwise
    invalid observations are removed before estimation; mismatched lengths are
    rejected rather than silently truncating one forecast series.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    model = np.asarray(loss_model, dtype=float).reshape(-1)
    baseline = np.asarray(loss_baseline, dtype=float).reshape(-1)
    if model.shape != baseline.shape:
        raise ValueError("loss_model and loss_baseline must have equal length")
    d = model - baseline
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return {"n": n, "statistic": float("nan"), "p_value": float("nan")}
    centered = d - d.mean()
    if accumulation_scale < 1:
        raise ValueError("accumulation_scale must be at least 1")
    lag = max(0, min(max(horizon - 1, accumulation_scale), n - 1))
    long_run = float(np.dot(centered, centered) / n)
    for k in range(1, lag + 1):
        covariance = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - k / (lag + 1)
        long_run += 2 * weight * covariance
    if long_run <= 0:
        return {"n": n, "statistic": float("nan"), "p_value": float("nan")}
    statistic = d.mean() / np.sqrt(long_run / n)
    correction = np.sqrt(
        max((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n, 0)
    )
    statistic *= correction
    # Use a finite-sample Student-t reference distribution.
    p_value = float(2 * stats.t.sf(abs(statistic), df=n - 1))
    return {
        "n": n,
        "effective_n": effective_sample_size(d),
        "hac_bandwidth": lag,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "mean_loss_difference": float(d.mean()),
    }


def paired_block_bootstrap_rmse_ci(
    prediction, baseline, target, groups, n_boot: int = 2000, seed: int = 123
) -> dict:
    """Bootstrap paired RMSE difference by site-year blocks to reduce overlap leakage."""
    p, b, y = map(lambda x: np.asarray(x, dtype=float), (prediction, baseline, target))
    groups = np.asarray(groups)
    valid = np.isfinite(p) & np.isfinite(b) & np.isfinite(y)
    p, b, y, groups = p[valid], b[valid], y[valid], groups[valid]
    unique = np.unique(groups)
    if len(unique) < 2:
        return {
            "n_blocks": len(unique),
            "rmse_difference_ci95": [float("nan"), float("nan")],
        }
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[g] for g in chosen])
        diffs[i] = np.sqrt(np.mean((p[rows] - y[rows]) ** 2)) - np.sqrt(
            np.mean((b[rows] - y[rows]) ** 2)
        )
    return {
        "n_blocks": len(unique),
        "rmse_difference_ci95": np.quantile(diffs, [0.025, 0.975]).tolist(),
    }


def count_drought_episodes(
    metadata, targets, mask, scale_index: int, threshold: float = -1.0
) -> int:
    """Count independent per-site drought runs in chronological target records."""
    episodes = 0
    grouped = {}
    for i, item in enumerate(metadata):
        grouped.setdefault(item["site"], []).append(i)
    for rows in grouped.values():
        records = {}
        for i in rows:
            for lead, date in enumerate(metadata[i]["target_dates"]):
                if mask[i, scale_index, lead] and np.isfinite(
                    targets[i, scale_index, lead]
                ):
                    month = date[:7]
                    records[month] = min(
                        records.get(month, float("inf")),
                        float(targets[i, scale_index, lead]),
                    )
        previous_month = None
        in_episode = False
        for month, value in sorted(records.items()):
            year, mon = map(int, month.split("-"))
            ordinal = year * 12 + mon
            contiguous = previous_month is not None and ordinal == previous_month + 1
            drought = value <= threshold
            if drought and (not in_episode or not contiguous):
                episodes += 1
            in_episode = drought
            previous_month = ordinal
    return episodes
