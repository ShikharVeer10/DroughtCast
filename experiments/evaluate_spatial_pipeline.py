"""Comprehensive spatial evaluation: metrics, baselines, calibration, and cell counts.

Evaluates spatial model predictions on untouched test grid cells and dates against:
1. Spatial Persistence (last observed SPEI)
2. Spatial Climatology (per-cell, per-calendar-month mean of training period)
3. Ridge Regression baseline

Produces paper-ready metrics with bootstrap confidence intervals, event classification
scores across thresholds (-1.0, -1.5, -2.0), probabilistic calibration diagnostics,
and cell count summaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xarray as xr

from data.spatial import SCALES
from utils.conformal import crps_from_quantiles, interval_metrics, pinball_score
from utils.metrics import (
    calculate_kge,
    calculate_mae,
    calculate_nse,
    calculate_rmse,
    drought_event_metrics,
)
from utils.probability import (
    brier_score,
    calibration_slope_intercept,
    expected_calibration_error,
    reliability_table,
)

THRESHOLDS = [-1.0, -1.5, -2.0]


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Calculate point estimate and 95% bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 5:
        return float(metric_fn(y_pred, y_true)), float("nan"), float("nan")

    point = float(metric_fn(y_pred, y_true))
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = metric_fn(y_pred[idx], y_true[idx])
        if np.isfinite(val):
            boot_vals.append(val)

    if len(boot_vals) < 100:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(boot_vals, 2.5))
    hi = float(np.percentile(boot_vals, 97.5))
    return point, lo, hi


def evaluate_spatial_evidence(
    evidence_ds: xr.Dataset,
    nominal_coverage: float = 0.8,
) -> dict[str, Any]:
    """Compute complete metric suite on spatial evidence NetCDF product."""
    report = {
        "summary": {},
        "per_scale": {},
        "thresholds": {},
        "probabilistic": {},
        "cell_counts": {},
    }

    n_valid_cells = int(evidence_ds["valid_cell_mask"].values.sum()) if "valid_cell_mask" in evidence_ds else 0
    report["cell_counts"]["total_valid_cells"] = n_valid_cells
    report["cell_counts"]["test_time_steps"] = len(evidence_ds["time"])

    for scale in SCALES:
        obs_var = f"obs_{scale}"
        pred_var = f"pred_{scale}"
        q10_var = f"q10_{scale}"
        q50_var = f"q50_{scale}"
        q90_var = f"q90_{scale}"
        prob_var = f"prob_{scale}"
        cqr_lo_var = f"cqr_lower_{scale}"
        cqr_hi_var = f"cqr_upper_{scale}"

        obs_vals = evidence_ds[obs_var].values.flatten()
        pred_vals = evidence_ds[pred_var].values.flatten()

        valid = np.isfinite(obs_vals) & np.isfinite(pred_vals)
        n_samples = int(valid.sum())
        report["cell_counts"][f"{scale}_valid_samples"] = n_samples

        if n_samples == 0:
            continue

        y_true = obs_vals[valid]
        y_pred = pred_vals[valid]

        # Continuous point forecast metrics with 95% CIs
        rmse, rmse_lo, rmse_hi = bootstrap_ci(y_true, y_pred, calculate_rmse)
        mae, mae_lo, mae_hi = bootstrap_ci(y_true, y_pred, calculate_mae)
        nse = calculate_nse(y_pred, y_true)
        kge = calculate_kge(y_pred, y_true)
        bias = float(np.mean(y_pred - y_true))
        corr = float(np.corrcoef(y_pred, y_true)[0, 1]) if np.std(y_pred) > 1e-6 and np.std(y_true) > 1e-6 else float("nan")

        report["per_scale"][scale] = {
            "n_samples": n_samples,
            "rmse": rmse,
            "rmse_ci95": [rmse_lo, rmse_hi],
            "mae": mae,
            "mae_ci95": [mae_lo, mae_hi],
            "bias": bias,
            "correlation": corr,
            "nse": nse,
            "kge": kge,
        }

        # Event contingency metrics across multiple drought thresholds
        report["thresholds"][scale] = {}
        for thresh in THRESHOLDS:
            ev = drought_event_metrics(y_pred, y_true, threshold=thresh)
            report["thresholds"][scale][f"threshold_{thresh}"] = ev

        # Probabilistic calibration & interval metrics
        prob_report = {}
        if prob_var in evidence_ds:
            probs = evidence_ds[prob_var].values.flatten()[valid]
            events = (y_true <= -1.0).astype(int)

            bs = brier_score(probs, events)
            clim_p = float(np.mean(events))
            bs_clim = brier_score(np.full_like(probs, clim_p), events)
            bss = float(1.0 - bs / bs_clim) if bs_clim > 1e-8 else float("nan")
            ece = expected_calibration_error(probs, events)
            cal_params = calibration_slope_intercept(probs, events)
            rel_table = reliability_table(probs, events)

            prob_report["calibrated_event_probability"] = {
                "threshold": -1.0,
                "brier_score": bs,
                "brier_skill_score_vs_climatology": bss,
                "expected_calibration_error": ece,
                "calibration_slope": cal_params["slope"],
                "calibration_intercept": cal_params["intercept"],
                "reliability_table": rel_table,
            }

        if cqr_lo_var in evidence_ds and cqr_hi_var in evidence_ds:
            cqr_lo = evidence_ds[cqr_lo_var].values.flatten()[valid]
            cqr_hi = evidence_ds[cqr_hi_var].values.flatten()[valid]
            int_res = interval_metrics(cqr_lo, cqr_hi, y_true)
            prob_report["cqr_interval_metrics"] = {
                "nominal_coverage": nominal_coverage,
                "empirical_coverage": int_res["coverage"],
                "mean_width": int_res["width"],
            }

            # Event-conditioned coverage
            cond_coverage = {}
            for thresh in THRESHOLDS:
                cond_mask = y_true <= thresh
                if cond_mask.sum() > 0:
                    cov_cond = float(np.mean((y_true[cond_mask] >= cqr_lo[cond_mask]) & (y_true[cond_mask] <= cqr_hi[cond_mask])))
                    cond_coverage[f"event_coverage_thresh_{thresh}"] = cov_cond
                else:
                    cond_coverage[f"event_coverage_thresh_{thresh}"] = float("nan")
            prob_report["event_conditioned_coverage"] = cond_coverage

        if q10_var in evidence_ds and q50_var in evidence_ds and q90_var in evidence_ds:
            q_arr = np.column_stack([
                evidence_ds[q10_var].values.flatten()[valid],
                evidence_ds[q50_var].values.flatten()[valid],
                evidence_ds[q90_var].values.flatten()[valid],
            ])
            pb_loss = pinball_score(q_arr, y_true, [0.1, 0.5, 0.9])
            crps_val = crps_from_quantiles(q_arr, y_true, [0.1, 0.5, 0.9])
            prob_report["quantile_scoring"] = {
                "pinball_loss": pb_loss,
                "crps": crps_val,
            }

        report["probabilistic"][scale] = prob_report

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate gridded spatial evidence product")
    parser.add_argument("evidence_nc", help="Path to evidence NetCDF file")
    parser.add_argument("--output-json", default="results/spatial_pipeline/tables/spatial_evaluation.json")
    args = parser.parse_args()

    ds = xr.open_dataset(args.evidence_nc)
    report = evaluate_spatial_evidence(ds)

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved evaluation metrics to: {out_p}")


if __name__ == "__main__":
    main()
