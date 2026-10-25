"""Validation and metrics for gridded spatial drought evidence.

The spatial product must contain observed targets and model predictions on the
same latitude/longitude/time grid.  This module deliberately does not
interpolate point-site predictions into a raster.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

SCALES = ("spei_3", "spei_6", "spei_12")


@dataclass(frozen=True)
class SpatialGrid:
    dataset: xr.Dataset
    lat: str
    lon: str
    time: str


def _coord(ds: xr.Dataset, names: tuple[str, ...]) -> str:
    for name in names:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(f"Spatial dataset requires one of coordinates {names}")


def load_spatial_grid(path: str | Path, observed_prefix: str = "obs_",
                      predicted_prefix: str = "pred_") -> SpatialGrid:
    """Load and validate a NetCDF grid with observed and predicted SPEI."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Spatial grid not found: {path}")
    ds = xr.open_dataset(path)
    lat, lon, time = (_coord(ds, ("lat", "latitude")),
                      _coord(ds, ("lon", "longitude")),
                      _coord(ds, ("time", "date")))
    if ds[lat].ndim != 1 or ds[lon].ndim != 1:
        raise ValueError("Latitude and longitude coordinates must be one-dimensional")
    if len(ds[lat]) < 2 or len(ds[lon]) < 2:
        raise ValueError("Spatial grid must contain at least two cells per axis")
    if np.any(np.diff(ds[lat].values) == 0) or np.any(np.diff(ds[lon].values) == 0):
        raise ValueError("Spatial coordinates contain duplicates")
    required = [f"{observed_prefix}{scale}" for scale in SCALES]
    required += [f"{predicted_prefix}{scale}" for scale in SCALES]
    missing = [name for name in required if name not in ds.data_vars]
    if missing:
        raise ValueError(f"Spatial evidence requires observed and predicted fields: {missing}")
    for name in required:
        dims = set(ds[name].dims)
        if not {lat, lon, time}.issubset(dims):
            raise ValueError(f"{name} must have dimensions including {time}, {lat}, {lon}")
    return SpatialGrid(ds, lat, lon, time)


def spatial_metrics(grid: SpatialGrid, threshold: float = -1.0,
                    observed_prefix: str = "obs_", predicted_prefix: str = "pred_") -> dict:
    """Return finite-cell metrics and drought event scores by scale."""
    out = {}
    for scale in SCALES:
        observed = grid.dataset[f"{observed_prefix}{scale}"].values
        predicted = grid.dataset[f"{predicted_prefix}{scale}"].values
        valid = np.isfinite(observed) & np.isfinite(predicted)
        if not valid.any():
            raise ValueError(f"No finite spatial pairs for {scale}")
        error = predicted[valid] - observed[valid]
        obs_event, pred_event = observed[valid] <= threshold, predicted[valid] <= threshold
        tp = int(np.sum(obs_event & pred_event))
        fp = int(np.sum(~obs_event & pred_event))
        fn = int(np.sum(obs_event & ~pred_event))
        out[scale] = {
            "n": int(valid.sum()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
            "pod": float(tp / (tp + fn)) if tp + fn else float("nan"),
            "far": float(fp / (tp + fp)) if tp + fp else float("nan"),
            "csi": float(tp / (tp + fp + fn)) if tp + fp + fn else float("nan"),
            "threshold": threshold,
        }
    return out
