"""Date-aligned, split-safe window construction for site-month tables."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from data.dataset import DroughtResearchDataset
from data.preprocessing import ClimateNormalizer

TARGETS = ["spei_3", "spei_6", "spei_12"]
DEFAULT_FEATURES = ["PRECTOTCORR", "T2M", "T2MDEW", "WS10M", "ALLSKY_SFC_SW_DWN",
                   "oni", "dmi", "nao", "pdo", "lag_spei_3", "lag_spei_6", "lag_spei_12"]


def split_for_date(date: pd.Timestamp) -> str | None:
    date = pd.Timestamp(date)
    if date <= pd.Timestamp("2010-12-01"):
        return "train"
    if pd.Timestamp("2012-01-01") <= date <= pd.Timestamp("2015-12-01"):
        return "val"
    if date >= pd.Timestamp("2017-01-01"):
        return "test"
    return None


def load_site_table(path: str | Path, features: list[str] | None = None,
                    seq_len: int = 6, forecast_steps: int = 4,
                    metadata_path: str | Path | None = None,
                    holdout_regions: list[str] | None = None) -> dict[str, DroughtResearchDataset]:
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "site", *TARGETS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    features = features or [c for c in DEFAULT_FEATURES if c in frame.columns]
    if not features:
        raise ValueError("No input features found")
    if set(features).difference(frame.columns):
        raise ValueError(f"Missing configured features: {sorted(set(features).difference(frame.columns))}")
    frame = frame.sort_values(["site", "date"])
    holdout_regions = set(holdout_regions or [])
    unknown_regions = holdout_regions.difference(frame["region"].dropna().unique()) if "region" in frame else holdout_regions
    if unknown_regions:
        raise ValueError(f"Unknown holdout regions: {sorted(unknown_regions)}")
    global_normalizer = None
    if holdout_regions:
        if "region" not in frame:
            raise ValueError("Leave-region-out requires a region column")
        dates = pd.to_datetime(frame["date"])
        source_train = frame[(~frame["region"].isin(holdout_regions)) & (dates <= pd.Timestamp("2010-12-01"))]
        global_normalizer = ClimateNormalizer().fit(source_train[features].apply(pd.to_numeric, errors="coerce").to_numpy())
    normalizers: dict[str, Any] = {}
    windows = {name: {"X": [], "Y": [], "mask": [], "metadata": []} for name in ("train", "val", "test")}
    for site, rows in frame.groupby("site", sort=False):
        rows = rows.set_index("date").sort_index()
        expected = pd.date_range(rows.index.min(), rows.index.max(), freq="MS")
        if not rows.index.equals(expected):
            rows = rows.reindex(expected)
            rows.index.name = "date"
            rows["site"] = site
        split_rows = np.array([split_for_date(d) for d in rows.index], dtype=object)
        region = str(rows["region"].dropna().iloc[0]) if "region" in rows and rows["region"].notna().any() else "unknown"
        feature_frame = rows[features].apply(pd.to_numeric, errors="coerce")
        train_mask = split_rows == "train"
        # Per-feature train-only moments, ignoring missing values.
        normalizer = global_normalizer if global_normalizer is not None else ClimateNormalizer().fit(feature_frame.to_numpy()[train_mask])
        normalized = normalizer.transform(feature_frame.to_numpy())
        normalizers[str(site)] = normalizer.to_dict()
        targets = rows[TARGETS].to_numpy(dtype=float)
        dates = rows.index
        for start in range(len(rows) - seq_len - forecast_steps + 1):
            input_idx = np.arange(start, start + seq_len)
            target_idx = np.arange(start + seq_len, start + seq_len + forecast_steps)
            split_parts = split_rows[np.concatenate([input_idx, target_idx])]
            if split_parts[0] is None or not np.all(split_parts == split_parts[0]):
                continue
            split = str(split_parts[0])
            if holdout_regions:
                if (region in holdout_regions and split != "test") or (region not in holdout_regions and split == "test"):
                    continue
            x = normalized[input_idx]
            y = targets[target_idx].T  # [scale, lead]
            mask = np.isfinite(y)
            # Missing inputs invalidate the sample; missing targets are retained as masks.
            if not np.isfinite(x).all():
                continue
            windows[split]["X"].append(x.astype(np.float32))
            windows[split]["Y"].append(np.nan_to_num(y, nan=0.0).astype(np.float32))
            windows[split]["mask"].append(mask)
            last_spei = rows.iloc[input_idx[-1]][TARGETS].to_dict()
            windows[split]["metadata"].append({"dates": [str(d.date()) for d in dates[input_idx]],
                "site": str(site), "region": region, "last_observed_spei": last_spei,
                "target_dates": [str(d.date()) for d in dates[target_idx]]})
    if metadata_path:
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).write_text(json.dumps({"features": features, "normalizers": normalizers}, indent=2), encoding="utf-8")
    result = {}
    for name, entries in windows.items():
        x = np.asarray(entries["X"], dtype=np.float32).reshape(-1, seq_len, len(features))
        y = np.asarray(entries["Y"], dtype=np.float32).reshape(-1, len(TARGETS), forecast_steps)
        mask = np.asarray(entries["mask"], dtype=bool).reshape(-1, len(TARGETS), forecast_steps)
        result[name] = DroughtResearchDataset(torch.from_numpy(x), torch.from_numpy(y),
                                                torch.from_numpy(mask), entries["metadata"])
    return result


def load_era_5_and_spei_data(inputs_csv_path: str, targets_nc_path: str, seq_len: int = 6,
                             forecast_steps: int = 4):
    raise RuntimeError("Legacy length-truncating NetCDF loader removed. Build data/monthly_sites.csv with data/build_dataset.py and use load_site_table().")
