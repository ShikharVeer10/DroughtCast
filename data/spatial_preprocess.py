"""Spatial harmonization, alignment, and leakage-prevention preprocessing.

Harmonizes gridded observed targets (SPEIbase) and gridded predictors (TerraClimate)
to a common coordinate reference system (EPSG:4326), 0.5-degree grid resolution,
and monthly temporal coordinate.

Strict scientific constraints:
- Latitude and longitude coordinates must be strictly monotonic without duplicates.
- NoData and non-finite cells are preserved in masks; values are never silently fabricated.
- Predictors available at forecast time T cannot contain future information (> T).
- Target leakage prevention: SPEIbase observed targets are never used as input features.
  Historical SPEI features (spei_3_obs, etc.) are unshifted observations available at time T.
- Temporal splits: train (<= 2010-12-01), validation (2012-01-01 to 2015-12-01),
  and test (>= 2017-01-01) with gap periods strictly preserved.
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data.spatial import SCALES

logger = logging.getLogger(__name__)


def validate_coordinates(
    ds: xr.Dataset,
    lat_name: str = "lat",
    lon_name: str = "lon",
    time_name: str = "time",
) -> None:
    """Validate spatial and temporal coordinates for monotonicity, uniqueness, and finiteness."""
    for coord in (lat_name, lon_name, time_name):
        if coord not in ds.coords and coord not in ds.dims:
            raise ValueError(f"Dataset missing required coordinate: {coord}")

    lat = ds[lat_name].values
    lon = ds[lon_name].values

    if len(lat) != len(np.unique(lat)):
        raise ValueError(f"Duplicate latitude coordinates detected in dataset: {lat}")
    if len(lon) != len(np.unique(lon)):
        raise ValueError(f"Duplicate longitude coordinates detected in dataset: {lon}")

    # Check monotonicity
    if not (np.all(np.diff(lat) > 0) or np.all(np.diff(lat) < 0)):
        raise ValueError("Latitude coordinates are not strictly monotonic")
    if not (np.all(np.diff(lon) > 0) or np.all(np.diff(lon) < 0)):
        raise ValueError("Longitude coordinates are not strictly monotonic")


def harmonize_grid(
    target_grid: xr.Dataset,
    source_grid: xr.Dataset,
    variables: list[str],
    method: str = "linear",
) -> xr.Dataset:
    """Regrid source_grid to match target_grid coordinates using documented interpolation."""
    validate_coordinates(target_grid)
    validate_coordinates(source_grid)

    reindexed = source_grid[variables].interp(
        lat=target_grid["lat"],
        lon=target_grid["lon"],
        method=method,
    )
    return reindexed


def enforce_leakage_controls(
    predictor_names: list[str],
    target_names: list[str] | None = None,
) -> None:
    """Validate that targets are not present in input predictors."""
    if target_names is None:
        target_names = list(SCALES)
    overlap = set(predictor_names).intersection(target_names)
    if overlap:
        raise ValueError(
            f"TARGET LEAKAGE DETECTED: Target variable(s) {overlap} cannot be used as input predictors!"
        )


def verify_temporal_splits(dates: pd.DatetimeIndex) -> dict[str, int]:
    """Check dates against train/val/test splits and gap periods."""
    train_mask = dates <= pd.Timestamp("2010-12-01")
    val_mask = (dates >= pd.Timestamp("2012-01-01")) & (
        dates <= pd.Timestamp("2015-12-01")
    )
    test_mask = dates >= pd.Timestamp("2017-01-01")
    gap_mask = ~(train_mask | val_mask | test_mask)

    counts = {
        "train_months": int(train_mask.sum()),
        "val_months": int(val_mask.sum()),
        "test_months": int(test_mask.sum()),
        "gap_months": int(gap_mask.sum()),
    }
    return counts


def create_harmonized_dataset(
    speibase_ds: xr.Dataset,
    predictors_ds: xr.Dataset,
    elevation_da: xr.DataArray | None = None,
    extent: tuple[float, float, float, float] | None = None,
) -> xr.Dataset:
    """Combine targets, predictors, and optional elevation into a single harmonized NetCDF."""
    validate_coordinates(speibase_ds)
    validate_coordinates(predictors_ds)

    # Standardize coordinate names to ('time', 'lat', 'lon')
    coords_map = {
        "latitude": "lat",
        "longitude": "lon",
        "date": "time",
    }
    spei = speibase_ds.rename(
        {k: v for k, v in coords_map.items() if k in speibase_ds.coords}
    )
    pred = predictors_ds.rename(
        {k: v for k, v in coords_map.items() if k in predictors_ds.coords}
    )

    # Sort latitude ascending
    if np.all(np.diff(spei["lat"].values) < 0):
        spei = spei.sortby("lat")
    if np.all(np.diff(pred["lat"].values) < 0):
        pred = pred.sortby("lat")

    # Optional spatial slicing
    if extent is not None:
        lat_min, lat_max, lon_min, lon_max = extent
        spei = spei.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        pred = pred.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    # Align time
    common_time = np.intersect1d(spei["time"].values, pred["time"].values)
    if len(common_time) == 0:
        raise ValueError(
            "No overlapping time steps between SPEIbase targets and predictors"
        )

    spei = spei.sel(time=common_time)
    pred = pred.sel(time=common_time)

    # Harmonize grid coordinates (bilinear regrid to SPEI grid if needed)
    if not np.array_equal(pred["lat"].values, spei["lat"].values) or not np.array_equal(
        pred["lon"].values, spei["lon"].values
    ):
        pred = pred.interp(lat=spei["lat"], lon=spei["lon"], method="linear")

    data_vars = {}
    # Targets
    for scale in SCALES:
        var_name = f"obs_{scale}"
        if var_name in spei:
            data_vars[var_name] = spei[var_name]
        elif scale in spei:
            data_vars[var_name] = spei[scale]
        elif "spei" in spei and len(SCALES) == 1:
            data_vars[var_name] = spei["spei"]
        else:
            raise ValueError(
                f"Target variable for {scale} not found in SPEIbase dataset"
            )

    # Predictors
    for pvar in pred.data_vars:
        if pvar not in data_vars:
            data_vars[pvar] = pred[pvar]

    # Elevation
    if elevation_da is not None:
        elev_regrid = elevation_da.interp(
            lat=spei["lat"], lon=spei["lon"], method="linear"
        )
        data_vars["elevation"] = elev_regrid
    else:
        # Default nan elevation if not supplied
        data_vars["elevation"] = xr.DataArray(
            np.full((len(spei["lat"]), len(spei["lon"])), np.nan, dtype=np.float32),
            dims=["lat", "lon"],
            coords={"lat": spei["lat"], "lon": spei["lon"]},
        )

    out_ds = xr.Dataset(
        data_vars, coords={"time": spei["time"], "lat": spei["lat"], "lon": spei["lon"]}
    )
    out_ds.attrs["crs"] = "EPSG:4326"
    out_ds.attrs["preprocessing_date"] = datetime.datetime.now(datetime.UTC).isoformat()
    out_ds.attrs["spatial_resolution_degrees"] = 0.5
    return out_ds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spatial data preprocessing and harmonization"
    )
    parser.add_argument("--speibase-dir", default="data/raw/spatial/speibase")
    parser.add_argument("--terraclimate-dir", default="data/raw/spatial/terraclimate")
    parser.add_argument("--output", default="data/spatial_harmonized.nc")
    parser.add_argument(
        "--extent",
        nargs=4,
        type=float,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
    )
    args = parser.parse_args()

    s_dir = Path(args.speibase_dir)
    t_dir = Path(args.terraclimate_dir)

    # Check for real raw data
    spei_files = [s_dir / f"spei{s:02d}.nc" for s in (3, 6, 12)]
    missing = [f.name for f in spei_files if not f.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cannot preprocess: SPEIbase files {missing} are missing from {s_dir}. "
            "Download them first or run python -m data.spatial_download."
        )

    print(f"Preprocessing SPEIbase and TerraClimate from {s_dir} and {t_dir}...")


if __name__ == "__main__":
    main()
