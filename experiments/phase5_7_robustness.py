"""Dependence-aware development uncertainty calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def block_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per site-origin block."""
    keys = frame[["site", "forecast_origin"]].drop_duplicates()
    if keys.duplicated().any():
        raise ValueError("duplicate block keys")
    return keys.sort_values(["site", "forecast_origin"]).reset_index(drop=True)


def cluster_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    replicates: int = 2000,
    seed: int = 5701,
) -> np.ndarray:
    """Bootstrap site clusters and complete site-origin blocks."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    sites = np.array(sorted(frame["site"].unique()))
    if len(sites) < 2:
        raise ValueError("at least two sites are required")
    blocks = {
        (site, origin): group[["observed", "prediction"]].to_numpy(dtype=float)
        for (site, origin), group in frame.groupby(["site", "forecast_origin"], sort=False)
    }
    origins_by_site = {site: np.array(sorted(frame.loc[frame["site"].eq(site), "forecast_origin"].unique())) for site in sites}
    values = []
    for _ in range(replicates):
        chosen_sites = rng.choice(sites, len(sites), replace=True)
        parts = []
        for site in chosen_sites:
            parts.extend(blocks[(site, origin)] for origin in rng.choice(origins_by_site[site], len(origins_by_site[site]), replace=True))
        sample = np.concatenate(parts)
        error = sample[:, 1] - sample[:, 0]
        values.append(float(np.sqrt(np.mean(error**2)) if metric == "rmse" else np.mean(np.abs(error))))
    return np.asarray(values)


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (float("nan"), float("nan"))
    tail = (1.0 - level) / 2.0
    return tuple(float(value) for value in np.quantile(values, [tail, 1.0 - tail]))


def adjust_bh(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted values for an explicit family."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 1.0
    for index in range(len(p) - 1, -1, -1):
        rank = index + 1
        running = min(running, p[order[index]] * len(p) / rank)
        adjusted[order[index]] = running
    return adjusted.tolist()
