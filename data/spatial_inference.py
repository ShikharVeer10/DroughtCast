"""Continuous spatial inference and evidence product generation.

Runs the trained DroughtCast model on each valid grid cell independently,
generating multi-horizon (T+1..T+4), multi-scale (SPEI-3/6/12) forecasts with
calibrated probabilities, CQR prediction intervals, errors, and static elevation.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from data.spatial import SCALES
from data.spatial_dataset import SpatialGridDataset
from models.forecaster import DroughtCast
from utils.conformal import calibrate_cqr
from utils.probability import fit_isotonic_calibrator


def run_cell_inference(
    model: DroughtCast,
    dataset: SpatialGridDataset,
    device: str = "cpu",
) -> dict[str, Any]:
    """Execute forward passes cell-by-cell for validation and test splits."""
    model.eval()
    results = []

    has_quantiles = bool(model.quantiles)
    with torch.no_grad():
        for cell in dataset.cells:
            if len(cell.X_test) == 0:
                continue

            # Val predictions for conformal/probability calibration
            val_preds_dict = None
            if len(cell.X_val) > 0:
                x_val = torch.from_numpy(cell.X_val).to(device)
                preds_val, _ = model(x_val)
                # preds_val: dict of {scale: [B, 4, 3] or [B, 4]}
                val_preds_dict = {s: preds_val[s].cpu().numpy() for s in SCALES}

            # Test predictions
            x_test = torch.from_numpy(cell.X_test).to(device)
            preds_test, _ = model(x_test)
            test_preds_dict = {s: preds_test[s].cpu().numpy() for s in SCALES}

            results.append(
                {
                    "lat": cell.lat,
                    "lon": cell.lon,
                    "elevation": cell.elevation,
                    "val_preds": val_preds_dict,
                    "val_targets": cell.Y_val,
                    "val_mask": cell.mask_val,
                    "test_preds": test_preds_dict,
                    "test_targets": cell.Y_test,
                    "test_mask": cell.mask_test,
                    "test_dates": cell.test_dates,
                }
            )

    return {
        "cell_results": results,
        "has_quantiles": has_quantiles,
        "quantiles": model.quantiles or [0.5],
    }


def calibrate_and_conformalize(
    cell_results: list[dict],
    quantiles: list[float],
    threshold: float = -1.0,
    nominal_coverage: float = 0.8,
) -> tuple[dict[str, Any], list[dict]]:
    """Fit validation-only isotonic calibrator and validation-only CQR adjustments."""
    # Pool validation samples across cells to fit robust calibration
    val_calibrators = {}
    cqr_corrections = {}

    for s_idx, scale in enumerate(SCALES):
        val_probs_raw = []
        val_events = []
        val_preds_cqr = []
        val_tars_cqr = []
        val_masks_cqr = []

        for cr in cell_results:
            if cr["val_preds"] is None or len(cr["val_targets"]) == 0:
                continue
            v_pred = cr["val_preds"][scale]  # [N, 4, 3]
            v_tar = cr["val_targets"][:, s_idx, :]  # [N, 4]
            v_mask = cr["val_mask"][:, s_idx, :]  # [N, 4]

            # Quantile interpolation to derive raw event probability
            if v_pred.ndim == 3 and v_pred.shape[-1] >= 3:
                q10 = v_pred[:, :, 0]
                q50 = v_pred[:, :, 1]
                q90 = v_pred[:, :, 2]
                raw_prob = np.where(
                    threshold <= q10,
                    0.1,
                    np.where(
                        threshold >= q90,
                        0.9,
                        0.1 + 0.4 * (threshold - q10) / np.clip(q50 - q10, 1e-4, None),
                    ),
                )
                raw_prob = np.clip(raw_prob, 0.0, 1.0)
            else:
                raw_prob = np.where(v_pred <= threshold, 1.0, 0.0)

            valid = v_mask & np.isfinite(v_tar)
            if valid.any():
                val_probs_raw.append(raw_prob[valid])
                val_events.append((v_tar[valid] <= threshold).astype(int))

            val_preds_cqr.append(v_pred)
            val_tars_cqr.append(v_tar)
            val_masks_cqr.append(v_mask)

        # Fit isotonic regression on pooled validation samples
        if val_probs_raw and len(np.concatenate(val_events)) >= 10:
            pooled_p = np.concatenate(val_probs_raw)
            pooled_y = np.concatenate(val_events)
            if np.unique(pooled_y).size >= 2:
                val_calibrators[scale] = fit_isotonic_calibrator(pooled_p, pooled_y)

        # Fit CQR lead-wise
        if val_preds_cqr:
            pooled_preds = np.concatenate(val_preds_cqr, axis=0)
            pooled_tars = np.concatenate(val_tars_cqr, axis=0)
            pooled_masks = np.concatenate(val_masks_cqr, axis=0)
            if pooled_preds.ndim == 3:
                cqr_corrections[scale] = calibrate_cqr(
                    pooled_preds,
                    pooled_tars,
                    pooled_masks,
                    quantiles,
                    coverage=nominal_coverage,
                )

    return {
        "calibrators": val_calibrators,
        "cqr_corrections": cqr_corrections,
    }, cell_results


def build_evidence_netcdf(
    dataset: SpatialGridDataset,
    cell_results: list[dict],
    calibration_artifacts: dict,
    output_path: Path | str,
    target_lead: int = 1,
) -> xr.Dataset:
    """Compile validation and test spatial predictions into a NetCDF evidence raster."""
    lats = dataset.lats
    lons = dataset.lons

    # Find common unique test dates across cells
    all_dates = set()
    for cr in cell_results:
        all_dates.update(cr["test_dates"])
    test_dates = sorted(all_dates)

    if not test_dates:
        raise ValueError("No valid test dates found across spatial cells")

    n_times = len(test_dates)
    n_lats = len(lats)
    n_lons = len(lons)

    # Coordinate index lookup
    lat_map = {float(lat): i for i, lat in enumerate(lats)}
    lon_map = {float(lon): i for i, lon in enumerate(lons)}
    date_map = {d: i for i, d in enumerate(test_dates)}

    # Initialize raster grids
    shape_3d = (n_times, n_lats, n_lons)
    shape_2d = (n_lats, n_lons)

    elevation_grid = np.full(shape_2d, np.nan, dtype=np.float32)
    valid_mask_grid = np.zeros(shape_2d, dtype=bool)

    grids_obs = {s: np.full(shape_3d, np.nan, dtype=np.float32) for s in SCALES}
    grids_pred = {s: np.full(shape_3d, np.nan, dtype=np.float32) for s in SCALES}
    grids_q10 = {s: np.full(shape_3d, np.nan, dtype=np.float32) for s in SCALES}
    grids_q50 = {s: np.full(shape_3d, np.nan, dtype=np.float32) for s in SCALES}
    grids_q90 = {s: np.full(shape_3d, np.nan, dtype=np.float32) for s in SCALES}
    grids_cqr_lo = {s: np.full(shape_3d, np.nan, dtype=np.float32) for s in SCALES}
    grids_cqr_hi = {s: np.full(shape_3d, np.nan, dtype=np.float32) for s in SCALES}
    grids_prob = {s: np.full(shape_3d, np.nan, dtype=np.float32) for s in SCALES}
    grids_error = {s: np.full(shape_3d, np.nan, dtype=np.float32) for s in SCALES}

    lead_idx = target_lead - 1  # 0-indexed lead (e.g. T+1 is index 0)
    calibrators = calibration_artifacts.get("calibrators", {})
    cqr_corrections = calibration_artifacts.get("cqr_corrections", {})

    for cr in cell_results:
        lat_i = lat_map.get(cr["lat"])
        lon_i = lon_map.get(cr["lon"])
        if lat_i is None or lon_i is None:
            continue

        elevation_grid[lat_i, lon_i] = cr["elevation"]
        valid_mask_grid[lat_i, lon_i] = True

        for t_idx, d_str in enumerate(cr["test_dates"]):
            time_i = date_map.get(d_str)
            if time_i is None:
                continue

            for s_idx, scale in enumerate(SCALES):
                pred = cr["test_preds"][scale][t_idx]  # [leads, quantiles] or [leads]
                tar = cr["test_targets"][t_idx, s_idx, lead_idx]
                mask = cr["test_mask"][t_idx, s_idx, lead_idx]

                if not mask:
                    continue

                grids_obs[scale][time_i, lat_i, lon_i] = tar

                if pred.ndim == 2:  # [leads, quantiles]
                    q10_val = float(pred[lead_idx, 0])
                    q50_val = float(pred[lead_idx, 1])
                    q90_val = float(pred[lead_idx, 2])
                    point_val = q50_val

                    grids_q10[scale][time_i, lat_i, lon_i] = q10_val
                    grids_q50[scale][time_i, lat_i, lon_i] = q50_val
                    grids_q90[scale][time_i, lat_i, lon_i] = q90_val

                    # Raw risk below -1.0
                    if -1.0 <= q10_val:
                        raw_p = 0.1
                    elif -1.0 >= q90_val:
                        raw_p = 0.9
                    else:
                        raw_p = 0.1 + 0.4 * (-1.0 - q10_val) / max(
                            q50_val - q10_val, 1e-4
                        )
                    raw_p = float(np.clip(raw_p, 0.0, 1.0))

                    # Calibrated event probability
                    if scale in calibrators:
                        cal_p = float(calibrators[scale].predict([raw_p])[0])
                    else:
                        cal_p = raw_p
                    grids_prob[scale][time_i, lat_i, lon_i] = cal_p

                    # CQR intervals
                    if scale in cqr_corrections and np.isfinite(
                        cqr_corrections[scale][lead_idx]
                    ):
                        delta = max(0.0, float(cqr_corrections[scale][lead_idx]))
                        grids_cqr_lo[scale][time_i, lat_i, lon_i] = q10_val - delta
                        grids_cqr_hi[scale][time_i, lat_i, lon_i] = q90_val + delta
                    else:
                        grids_cqr_lo[scale][time_i, lat_i, lon_i] = q10_val
                        grids_cqr_hi[scale][time_i, lat_i, lon_i] = q90_val

                else:
                    point_val = float(pred[lead_idx])
                    grids_prob[scale][time_i, lat_i, lon_i] = (
                        1.0 if point_val <= -1.0 else 0.0
                    )

                grids_pred[scale][time_i, lat_i, lon_i] = point_val
                grids_error[scale][time_i, lat_i, lon_i] = point_val - tar

    # Assemble dataset
    data_vars = {
        "elevation": (("lat", "lon"), elevation_grid),
        "valid_cell_mask": (("lat", "lon"), valid_mask_grid),
    }

    for scale in SCALES:
        data_vars[f"obs_{scale}"] = (("time", "lat", "lon"), grids_obs[scale])
        data_vars[f"pred_{scale}"] = (("time", "lat", "lon"), grids_pred[scale])
        data_vars[f"q10_{scale}"] = (("time", "lat", "lon"), grids_q10[scale])
        data_vars[f"q50_{scale}"] = (("time", "lat", "lon"), grids_q50[scale])
        data_vars[f"q90_{scale}"] = (("time", "lat", "lon"), grids_q90[scale])
        data_vars[f"cqr_lower_{scale}"] = (("time", "lat", "lon"), grids_cqr_lo[scale])
        data_vars[f"cqr_upper_{scale}"] = (("time", "lat", "lon"), grids_cqr_hi[scale])
        data_vars[f"prob_{scale}"] = (("time", "lat", "lon"), grids_prob[scale])
        data_vars[f"error_{scale}"] = (("time", "lat", "lon"), grids_error[scale])

    coords = {
        "time": pd.to_datetime(test_dates),
        "lat": lats,
        "lon": lons,
    }

    evidence_ds = xr.Dataset(data_vars, coords=coords)
    evidence_ds.attrs["target_lead"] = f"T+{target_lead}"
    evidence_ds.attrs["crs"] = "EPSG:4326"
    evidence_ds.attrs["creation_date"] = datetime.datetime.now(datetime.UTC).isoformat()
    evidence_ds.attrs["description"] = (
        "DroughtCast gridded spatial evidence product containing observed SPEIbase, "
        "model predicted SPEI, quantiles, calibrated probabilities, CQR uncertainty intervals, "
        "and elevation on a common 0.5-degree grid."
    )

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    evidence_ds.to_netcdf(out_p)
    return evidence_ds
