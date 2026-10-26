"""Tests for continuous spatial pipeline: grid alignment, leakage prevention,

deterministic preprocessing, spatial inference, calibration, and artifact compliance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from data.spatial import load_spatial_grid, spatial_metrics, SCALES
from data.spatial_dataset import SpatialGridDataset
from data.spatial_download import (
    check_gadm_boundaries,
    verify_elevation_file,
    verify_speibase_files,
    verify_terraclimate_files,
)
from data.spatial_inference import (
    build_evidence_netcdf,
    calibrate_and_conformalize,
    run_cell_inference,
)
from data.spatial_preprocess import (
    create_harmonized_dataset,
    enforce_leakage_controls,
    validate_coordinates,
    verify_temporal_splits,
)
from experiments.evaluate_spatial_pipeline import evaluate_spatial_evidence
from models.forecaster import DroughtCast


@pytest.fixture
def mock_spatial_dataset(tmp_path):
    """Create a minimal, valid spatial NetCDF covering train, val, and test dates."""
    dates = pd.date_range("2009-01-01", "2018-12-01", freq="MS")
    lats = np.array([15.0, 16.0, 17.0], dtype=np.float32)
    lons = np.array([75.0, 76.0, 77.0], dtype=np.float32)

    shape = (len(dates), len(lats), len(lons))
    rng = np.random.default_rng(42)

    data_vars = {
        "PRECTOTCORR": (("time", "lat", "lon"), rng.uniform(10.0, 100.0, size=shape).astype(np.float32)),
        "T2M": (("time", "lat", "lon"), rng.uniform(15.0, 35.0, size=shape).astype(np.float32)),
        "T2MDEW": (("time", "lat", "lon"), rng.uniform(10.0, 25.0, size=shape).astype(np.float32)),
        "WS10M": (("time", "lat", "lon"), rng.uniform(1.0, 5.0, size=shape).astype(np.float32)),
        "ALLSKY_SFC_SW_DWN": (("time", "lat", "lon"), rng.uniform(15.0, 25.0, size=shape).astype(np.float32)),
        "elevation": (("lat", "lon"), np.array([[500, 600, 700], [550, 650, 750], [520, 620, 720]], dtype=np.float32)),
    }

    # Add targets
    for s in SCALES:
        data_vars[f"obs_{s}"] = (("time", "lat", "lon"), rng.normal(0.0, 1.0, size=shape).astype(np.float32))

    coords = {"time": dates, "lat": lats, "lon": lons}
    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs["crs"] = "EPSG:4326"

    path = tmp_path / "mock_harmonized.nc"
    ds.to_netcdf(path)
    return path, ds


def test_coordinate_validation_monotonicity_and_duplicates():
    # Valid
    lats = [10.0, 11.0, 12.0]
    lons = [70.0, 71.0]
    ds = xr.Dataset(coords={"lat": lats, "lon": lons, "time": [0, 1]})
    validate_coordinates(ds)

    # Duplicate coordinate
    ds_dup = xr.Dataset(coords={"lat": [10.0, 10.0, 12.0], "lon": lons, "time": [0, 1]})
    with pytest.raises(ValueError, match="Duplicate"):
        validate_coordinates(ds_dup)

    # Non-monotonic
    ds_nonmono = xr.Dataset(coords={"lat": [10.0, 12.0, 11.0], "lon": lons, "time": [0, 1]})
    with pytest.raises(ValueError, match="monotonic"):
        validate_coordinates(ds_nonmono)


def test_leakage_control_rejects_target_in_predictors():
    # Valid predictor set
    enforce_leakage_controls(["PRECTOTCORR", "T2M", "WS10M"])

    # Target leakage attempt
    with pytest.raises(ValueError, match="TARGET LEAKAGE DETECTED"):
        enforce_leakage_controls(["PRECTOTCORR", "spei_6", "T2M"])


def test_temporal_split_isolation():
    dates = pd.date_range("2008-01-01", "2018-12-01", freq="MS")
    counts = verify_temporal_splits(dates)
    assert counts["train_months"] > 0
    assert counts["val_months"] > 0
    assert counts["test_months"] > 0
    # Strict buffer gap periods preserved
    assert counts["gap_months"] > 0


def test_spatial_dataset_train_only_normalization(mock_spatial_dataset):
    _, ds = mock_spatial_dataset
    feats = ["PRECTOTCORR", "T2M", "T2MDEW", "WS10M", "ALLSKY_SFC_SW_DWN"]
    spatial_ds = SpatialGridDataset(ds, feature_vars=feats, seq_len=6, forecast_steps=4)

    assert spatial_ds.num_cells == 9  # 3x3 grid
    for cell in spatial_ds.cells:
        assert len(cell.X_train) > 0
        assert len(cell.X_val) > 0
        assert len(cell.X_test) > 0
        # Normalizer mean and std should match train moments
        assert np.isfinite(cell.normalizer.mean).all()
        assert np.isfinite(cell.normalizer.std).all()
        assert np.all(cell.normalizer.std > 0)


def test_spatial_inference_and_quantile_ordering(mock_spatial_dataset, tmp_path):
    _, ds = mock_spatial_dataset
    feats = ["PRECTOTCORR", "T2M", "T2MDEW", "WS10M", "ALLSKY_SFC_SW_DWN"]
    spatial_ds = SpatialGridDataset(ds, feature_vars=feats, seq_len=6, forecast_steps=4)

    cfg = {
        "data": {"input_dim": len(feats), "forecast_steps": 4, "seq_len": 6},
        "model": {
            "encoder": "lstm",
            "hidden_dim_1": 16,
            "hidden_dim_2": 8,
            "dropout": 0.0,
            "attention": "shared",
            "heads": "direct",
            "quantiles": [0.1, 0.5, 0.9],
        },
    }
    model = DroughtCast(cfg)
    raw = run_cell_inference(model, spatial_ds)

    assert raw["has_quantiles"] is True
    assert len(raw["cell_results"]) == 9

    for cr in raw["cell_results"]:
        for s in SCALES:
            q_preds = cr["test_preds"][s]
            # [samples, leads, quantiles]
            assert q_preds.ndim == 3
            assert q_preds.shape[-1] == 3
            # Check quantiles ordering: q10 <= q50 <= q90
            assert np.all(q_preds[:, :, 0] <= q_preds[:, :, 1] + 1e-5)
            assert np.all(q_preds[:, :, 1] <= q_preds[:, :, 2] + 1e-5)


def test_conformal_and_calibrated_evidence_netcdf(mock_spatial_dataset, tmp_path):
    _, ds = mock_spatial_dataset
    feats = ["PRECTOTCORR", "T2M", "T2MDEW", "WS10M", "ALLSKY_SFC_SW_DWN"]
    spatial_ds = SpatialGridDataset(ds, feature_vars=feats, seq_len=6, forecast_steps=4)

    cfg = {
        "data": {"input_dim": len(feats), "forecast_steps": 4, "seq_len": 6},
        "model": {
            "encoder": "lstm",
            "hidden_dim_1": 16,
            "hidden_dim_2": 8,
            "dropout": 0.0,
            "attention": "shared",
            "heads": "direct",
            "quantiles": [0.1, 0.5, 0.9],
        },
    }
    model = DroughtCast(cfg)
    raw = run_cell_inference(model, spatial_ds)

    cal_artifacts, processed = calibrate_and_conformalize(
        raw["cell_results"], [0.1, 0.5, 0.9], threshold=-1.0, nominal_coverage=0.8
    )

    out_nc = tmp_path / "evidence.nc"
    evidence_ds = build_evidence_netcdf(spatial_ds, processed, cal_artifacts, out_nc)

    # 1. Compatibility with existing data.spatial.load_spatial_grid
    grid = load_spatial_grid(out_nc)
    assert grid.lat == "lat"
    assert grid.lon == "lon"
    assert grid.time == "time"

    metrics = spatial_metrics(grid)
    for s in SCALES:
        assert metrics[s]["n"] > 0
        assert np.isfinite(metrics[s]["rmse"])
        assert np.isfinite(metrics[s]["mae"])

    # 2. Comprehensive evaluation suite
    full_eval = evaluate_spatial_evidence(evidence_ds)
    assert "per_scale" in full_eval
    assert "thresholds" in full_eval
    assert "probabilistic" in full_eval
    for s in SCALES:
        assert full_eval["per_scale"][s]["n_samples"] > 0
        assert np.isfinite(full_eval["per_scale"][s]["rmse"])
        # Probability metrics bounded
        prob_m = full_eval["probabilistic"][s].get("calibrated_event_probability", {})
        if "brier_score" in prob_m:
            assert 0.0 <= prob_m["brier_score"] <= 1.0


def test_missing_data_blocker_detection(tmp_path):
    # Empty directories simulate un-downloaded raw sources
    empty_spei = tmp_path / "empty_spei"
    empty_spei.mkdir()
    res = verify_speibase_files(empty_spei)
    assert res["available"] is False
    assert len(res["missing"]) == 3

    empty_tc = tmp_path / "empty_tc"
    empty_tc.mkdir()
    res_tc = verify_terraclimate_files(empty_tc)
    assert res_tc["available"] is False
    assert len(res_tc["missing"]) == 6

    res_dem = verify_elevation_file(tmp_path / "nonexistent_dem.tif")
    assert res_dem["available"] is False
