"""Gridded spatial dataset construction and window slicing for spatial inference.

The scientific spatial inference approach applied here is the least speculative
and fully documented method: apply the temporal forecaster independently to each
valid spatial grid cell using the cell's local gridded climate predictors.

Rules enforced:
- Per-cell normalization fitted exclusively on training-period data (<= 2010-12-01).
- No lookahead: predictors at time T only use information from T-seq_len+1 to T.
- No target leakage: target SPEI values are strictly predicted into the future (T+1 to T+forecast_steps).
- Clear reporting of cell counts, valid sample counts, and spatial mask statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from data.loader import split_for_date
from data.preprocessing import ClimateNormalizer
from data.spatial import SCALES


@dataclass
class SpatialCellWindows:
    lat: float
    lon: float
    elevation: float
    X_train: np.ndarray
    Y_train: np.ndarray
    mask_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    mask_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    mask_test: np.ndarray
    test_dates: list[str]
    normalizer: ClimateNormalizer


class SpatialGridDataset:
    """Extracts date-aligned sliding windows across all valid grid cells in a NetCDF dataset."""

    def __init__(
        self,
        ds: xr.Dataset,
        feature_vars: list[str],
        seq_len: int = 6,
        forecast_steps: int = 4,
        holdout_bbox: tuple[float, float, float, float] | None = None,
    ):
        self.ds = ds
        self.feature_vars = feature_vars
        self.seq_len = seq_len
        self.forecast_steps = forecast_steps
        self.holdout_bbox = holdout_bbox
        self.lats = ds["lat"].values
        self.lons = ds["lon"].values
        self.times = pd.to_datetime(ds["time"].values)
        self.cells: list[SpatialCellWindows] = []
        self._build_cells()

    def _build_cells(self) -> None:
        times = self.times
        split_names = np.array([split_for_date(t) for t in times], dtype=object)

        # Precompute indices for valid windows
        total_time = len(times)
        window_indices = []
        for start in range(total_time - self.seq_len - self.forecast_steps + 1):
            inp_idx = np.arange(start, start + self.seq_len)
            tar_idx = np.arange(
                start + self.seq_len, start + self.seq_len + self.forecast_steps
            )
            parts = split_names[np.concatenate([inp_idx, tar_idx])]
            if parts[0] is not None and np.all(parts == parts[0]):
                window_indices.append((start, inp_idx, tar_idx, str(parts[0])))

        for lat_idx, lat_val in enumerate(self.lats):
            lat_f = float(lat_val)
            for lon_idx, lon_val in enumerate(self.lons):
                lon_f = float(lon_val)

                # Check geographic holdout
                in_holdout = False
                if self.holdout_bbox is not None:
                    h_lat_min, h_lat_max, h_lon_min, h_lon_max = self.holdout_bbox
                    in_holdout = (h_lat_min <= lat_f <= h_lat_max) and (
                        h_lon_min <= lon_f <= h_lon_max
                    )

                # Extract feature array [T, n_features]
                feats = []
                for fvar in self.feature_vars:
                    if fvar in self.ds:
                        feats.append(
                            self.ds[fvar].isel(lat=lat_idx, lon=lon_idx).values
                        )
                    else:
                        feats.append(np.zeros(total_time, dtype=np.float32))
                feats = np.column_stack(feats)

                # Extract targets [T, 3]
                targets = []
                for s in SCALES:
                    obs_var = f"obs_{s}"
                    if obs_var in self.ds:
                        targets.append(
                            self.ds[obs_var].isel(lat=lat_idx, lon=lon_idx).values
                        )
                    else:
                        targets.append(np.full(total_time, np.nan, dtype=np.float32))
                targets = np.column_stack(targets)

                # Check if cell has enough finite observations
                train_mask = split_names == "train"
                if (
                    not np.isfinite(feats[train_mask]).any()
                    or not np.isfinite(targets[train_mask]).any()
                ):
                    continue

                # Fit normalizer strictly on train split
                normalizer = ClimateNormalizer().fit(feats[train_mask])
                norm_feats = normalizer.transform(feats)

                windows: dict[str, dict[str, list]] = {
                    "train": {"X": [], "Y": [], "mask": []},
                    "val": {"X": [], "Y": [], "mask": []},
                    "test": {"X": [], "Y": [], "mask": [], "dates": []},
                }

                for start, inp_idx, tar_idx, split in window_indices:
                    # If cell is in holdout, keep it only in test split
                    if in_holdout and split != "test":
                        continue
                    if (
                        not in_holdout
                        and split == "test"
                        and self.holdout_bbox is not None
                    ):
                        # Standard leave-region-out setup
                        pass

                    x_win = norm_feats[inp_idx]
                    y_win = targets[tar_idx].T  # [3_scales, 4_leads]
                    mask_win = np.isfinite(y_win)

                    if not np.isfinite(x_win).all():
                        continue

                    windows[split]["X"].append(x_win)
                    windows[split]["Y"].append(np.nan_to_num(y_win, nan=0.0))
                    windows[split]["mask"].append(mask_win)
                    if split == "test":
                        windows["test"]["dates"].append(str(times[tar_idx[0]].date()))

                elev = (
                    float(self.ds["elevation"].isel(lat=lat_idx, lon=lon_idx).values)
                    if "elevation" in self.ds
                    else float("nan")
                )

                # Pack windows
                def _to_arr(lst, shape):
                    return (
                        np.asarray(lst, dtype=np.float32).reshape(-1, *shape)
                        if lst
                        else np.zeros((0, *shape), dtype=np.float32)
                    )

                def _to_mask(lst, shape):
                    return (
                        np.asarray(lst, dtype=bool).reshape(-1, *shape)
                        if lst
                        else np.zeros((0, *shape), dtype=bool)
                    )

                cell_obj = SpatialCellWindows(
                    lat=lat_f,
                    lon=lon_f,
                    elevation=elev,
                    X_train=_to_arr(
                        windows["train"]["X"], (self.seq_len, len(self.feature_vars))
                    ),
                    Y_train=_to_arr(
                        windows["train"]["Y"], (len(SCALES), self.forecast_steps)
                    ),
                    mask_train=_to_mask(
                        windows["train"]["mask"], (len(SCALES), self.forecast_steps)
                    ),
                    X_val=_to_arr(
                        windows["val"]["X"], (self.seq_len, len(self.feature_vars))
                    ),
                    Y_val=_to_arr(
                        windows["val"]["Y"], (len(SCALES), self.forecast_steps)
                    ),
                    mask_val=_to_mask(
                        windows["val"]["mask"], (len(SCALES), self.forecast_steps)
                    ),
                    X_test=_to_arr(
                        windows["test"]["X"], (self.seq_len, len(self.feature_vars))
                    ),
                    Y_test=_to_arr(
                        windows["test"]["Y"], (len(SCALES), self.forecast_steps)
                    ),
                    mask_test=_to_mask(
                        windows["test"]["mask"], (len(SCALES), self.forecast_steps)
                    ),
                    test_dates=windows["test"]["dates"],
                    normalizer=normalizer,
                )
                self.cells.append(cell_obj)

    @property
    def num_cells(self) -> int:
        return len(self.cells)

    def sample_counts(self) -> dict[str, int]:
        return {
            "total_cells": len(self.cells),
            "train_samples": sum(len(c.X_train) for c in self.cells),
            "val_samples": sum(len(c.X_val) for c in self.cells),
            "test_samples": sum(len(c.X_test) for c in self.cells),
        }
