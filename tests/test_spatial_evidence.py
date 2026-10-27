import numpy as np
import pytest
import xarray as xr

from data.spatial import load_spatial_grid, spatial_metrics


def _grid(path):
    coords = {
        "time": np.array(["2017-01-01", "2017-02-01"], dtype="datetime64[ns]"),
        "lat": [10.0, 11.0],
        "lon": [75.0, 76.0],
    }
    values = np.ones((2, 2, 2), dtype=float)
    ds = xr.Dataset(
        {
            f"{prefix}{scale}": (
                ("time", "lat", "lon"),
                values + (0.1 if prefix == "pred_" else 0),
            )
            for prefix in ("obs_", "pred_")
            for scale in ("spei_3", "spei_6", "spei_12")
        },
        coords=coords,
    )
    ds.to_netcdf(path)


def test_spatial_grid_metrics_require_observed_and_predicted_fields(tmp_path):
    path = tmp_path / "grid.nc"
    _grid(path)
    grid = load_spatial_grid(path)
    report = spatial_metrics(grid)
    assert report["spei_6"]["n"] == 8
    assert report["spei_6"]["rmse"] == pytest.approx(0.1)


def test_spatial_grid_rejects_missing_prediction(tmp_path):
    path = tmp_path / "grid.nc"
    _grid(path)
    ds = xr.open_dataset(path).drop_vars("pred_spei_12")
    ds.to_netcdf(path.with_name("missing.nc"))
    with pytest.raises(ValueError, match="pred_spei_12"):
        load_spatial_grid(path.with_name("missing.nc"))
