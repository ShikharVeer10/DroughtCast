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
from data.protocol import (
    DATASET_PROTOCOL_VERSION,
    INDEX_ALIGNMENT,
    SPLIT_BOUNDARIES,
    SPLITS,
)
from data.quality import validate_site_table

TARGETS = ["spei_3", "spei_6", "spei_12"]
DEFAULT_FEATURES = [
    "PRECTOTCORR",
    "T2M",
    "T2MDEW",
    "WS10M",
    "ALLSKY_SFC_SW_DWN",
    "oni",
    "dmi",
    "nao",
    "pdo",
    "spei_3_obs",
    "spei_6_obs",
    "spei_12_obs",
]
def validate_dataset_protocol(
    path: str | Path = "data/dataset_metadata.json", target_source: str | None = None
) -> dict:
    """Reject datasets built before the current leakage-alignment protocol."""
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Dataset provenance is missing at {metadata_path}. Rebuild the monthly dataset with data/build_dataset.py."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != DATASET_PROTOCOL_VERSION
        or metadata.get("index_alignment") != INDEX_ALIGNMENT
        or metadata.get("split_boundaries") != SPLIT_BOUNDARIES
        or (target_source is not None and metadata.get("target_source", "local") != target_source)
    ):
        raise RuntimeError(
            f"Dataset at {metadata_path} predates the current teleconnection alignment. "
            "Rebuild it with data/build_dataset.py before training or evaluation."
        )
    return metadata


def split_for_date(date: pd.Timestamp) -> str | None:
    return split_for_date_with_boundaries(date, **SPLIT_BOUNDARIES)


def split_for_date_with_boundaries(date: pd.Timestamp, train_end: str, val_start: str, val_end: str, test_start: str, test_end: str | None = None) -> str | None:
    date = pd.Timestamp(date)
    if date <= pd.Timestamp(train_end):
        return "train"
    if pd.Timestamp(val_start) <= date <= pd.Timestamp(val_end):
        return "val"
    if date >= pd.Timestamp(test_start) and (test_end is None or date <= pd.Timestamp(test_end)):
        return "test"
    return None


def load_site_table(
    path: str | Path,
    features: list[str] | None = None,
    seq_len: int = 6,
    forecast_steps: int = 4,
    metadata_path: str | Path | None = None,
    holdout_regions: list[str] | None = None,
    site_filter: list[str] | None = None,
    input_extras: list[str] | None = None,
    static_context: bool = False,
    residual_heads: bool = False,
    site_specs: dict[str, dict[str, Any]] | None = None,
    split_config: dict[str, str] | None = None,
) -> dict[str, DroughtResearchDataset]:
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame.replace([-999.0, -99.0, -9999.0, -32768.0], np.nan)
    frame["site"] = frame["site"].astype(str).str.strip().str.lower()
    if site_filter is not None:
        allowed_sites = {str(s).lower() for s in site_filter}
        unexpected = set(frame["site"].unique()) - allowed_sites
        if unexpected:
            raise ValueError(f"Non-Maharashtra/unconfigured sites in dataset: {sorted(unexpected)}")
    required = {"date", "site", *TARGETS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    allowed_regions = {"Marathwada", "Vidarbha", "Madhya_Maharashtra", "Konkan"}
    if "region" in frame and site_filter and not set(frame["region"].dropna().astype(str).str.replace(" ", "_", regex=False).unique()).issubset(allowed_regions):
        raise ValueError("Input contains non-Maharashtra regions")
    input_extras = list(input_extras or [])
    known_extras = {"month_sincos", "water_balance"}
    unknown_extras = set(input_extras) - known_extras
    if unknown_extras:
        raise ValueError(f"Unknown input extras: {sorted(unknown_extras)}")
    if "month_sincos" in input_extras:
        month = frame["date"].dt.month.to_numpy()
        frame["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
        frame["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    if "water_balance" in input_extras:
        if not {"PRECTOTCORR", "pet_mm"}.issubset(frame.columns):
            raise ValueError("water_balance requires PRECTOTCORR and pet_mm columns")
        frame["water_balance"] = pd.to_numeric(
            frame["PRECTOTCORR"], errors="coerce"
        ) * frame["date"].dt.days_in_month - pd.to_numeric(
            frame["pet_mm"], errors="coerce"
        )
    if "region" in frame:
        region_alias = {"madhya maharashtra": "Madhya_Maharashtra"}
        frame["region"] = frame["region"].astype(str).replace(region_alias)
    features = list(features or [c for c in DEFAULT_FEATURES if c in frame.columns])
    if "month_sincos" in input_extras:
        features.extend(["month_sin", "month_cos"])
    if "water_balance" in input_extras:
        features.append("water_balance")
    if not features:
        raise ValueError("No input features found")
    if set(features).difference(frame.columns):
        raise ValueError(
            f"Missing configured features: {sorted(set(features).difference(frame.columns))}"
        )
    frame = frame.sort_values(["site", "date"])
    if site_filter:
        unknown_sites = set(site_filter) - set(frame["site"].astype(str).unique())
        if unknown_sites:
            raise ValueError(f"Unknown configured site(s): {sorted(unknown_sites)}")
        frame = frame[frame["site"].astype(str).isin(site_filter)].copy()
    validate_site_table(frame, features, site_filter, strict=site_filter is not None)
    holdout_regions = set(holdout_regions or [])
    unknown_regions = (
        holdout_regions.difference(frame["region"].dropna().unique())
        if "region" in frame
        else holdout_regions
    )
    if unknown_regions:
        raise ValueError(f"Unknown holdout regions: {sorted(unknown_regions)}")
    site_specs = dict(site_specs or {})
    if metadata_path and Path(metadata_path).is_file():
        try:
            saved_metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            site_specs.update(saved_metadata.get("sites", {}))
        except (OSError, json.JSONDecodeError):
            site_specs = {}
    global_normalizer = None
    if holdout_regions:
        if "region" not in frame:
            raise ValueError("Leave-region-out requires a region column")
        dates = pd.to_datetime(frame["date"])
        source_train = frame[
            (~frame["region"].isin(holdout_regions))
            & (dates <= pd.Timestamp(SPLITS["train_end"]))
        ]
        global_normalizer = ClimateNormalizer().fit(
            source_train[features].apply(pd.to_numeric, errors="coerce").to_numpy()
        )
    normalizers: dict[str, Any] = {}
    windows = {
        name: {
            "X": [],
            "Y": [],
            "mask": [],
            "metadata": [],
            "last_spei": [],
            "static": [],
        }
        for name in ("train", "val", "test")
    }
    for site, rows in frame.groupby("site", sort=False):
        rows = rows.set_index("date").sort_index()
        if rows.index.has_duplicates:
            raise ValueError(f"Duplicate monthly observations for site {site}")
        expected = pd.date_range(rows.index.min(), rows.index.max(), freq="MS")
        if not rows.index.equals(expected):
            print(f"Missing climate inputs before windowing: site={site}, missing_months={len(expected.difference(rows.index))}")
            rows = rows.reindex(expected)
            rows.index.name = "date"
            rows["site"] = site
        split_fn = split_for_date
        if split_config:
            split_fn = lambda d: split_for_date_with_boundaries(
                d, split_config["train_end"], split_config["val_start"],
                split_config["val_end"], split_config["test_start"],
                split_config.get("test_end")
            )
        split_rows = np.array([split_fn(d) for d in rows.index], dtype=object)
        region = (
            str(rows["region"].dropna().iloc[0])
            if "region" in rows and rows["region"].notna().any()
            else "unknown"
        )
        # Do not interpolate across chronological boundaries: that would use
        # validation/test observations to impute an earlier split. Missing
        # inputs remain invalid windows rather than receiving a future-derived
        # value; the normalizer itself is still fitted on train rows only.
        feature_frame = rows[features].apply(pd.to_numeric, errors="coerce")
        train_mask = split_rows == "train"
        missing_counts = feature_frame.isna().sum()
        if missing_counts.any():
            print(f"Missing climate inputs before windowing: site={site}, " + ", ".join(f"{k}={int(v)}" for k,v in missing_counts.items() if v))
        # Per-feature train-only moments, ignoring missing values.
        if not train_mask.any():
            raise ValueError(f"Site {site} has no training observations")
        normalizer = (
            global_normalizer
            if global_normalizer is not None
            else ClimateNormalizer().fit(feature_frame.to_numpy()[train_mask])
        )
        normalized = normalizer.transform(feature_frame.to_numpy())
        normalized[~np.isfinite(normalized)] = np.nan
        normalizers[str(site)] = normalizer.to_dict()
        targets = rows[TARGETS].to_numpy(dtype=float)
        dates = rows.index
        spec = site_specs.get(str(site), {})
        latitude = float(spec["lat"]) if "lat" in spec else 0.0
        precipitation = (
            pd.to_numeric(rows["PRECTOTCORR"], errors="coerce")
            if "PRECTOTCORR" in rows
            else None
        )
        pet = (
            pd.to_numeric(rows["pet_mm"], errors="coerce") if "pet_mm" in rows else None
        )
        train_precip = (
            precipitation[train_mask].mean() if precipitation is not None else np.nan
        )
        train_pet = pet[train_mask].mean() if pet is not None else np.nan
        aridity = (
            float(train_precip / train_pet)
            if (np.isfinite(train_precip) and np.isfinite(train_pet) and train_pet != 0)
            else 0.0
        )
        for start in range(len(rows) - seq_len - forecast_steps + 1):
            input_idx = np.arange(start, start + seq_len)
            target_idx = np.arange(start + seq_len, start + seq_len + forecast_steps)
            split_parts = split_rows[np.concatenate([input_idx, target_idx])]
            if split_parts[0] is None or not np.all(split_parts == split_parts[0]):
                continue
            split = str(split_parts[0])
            if holdout_regions and (
                (region in holdout_regions and split != "test")
                or (region not in holdout_regions and split == "test")
            ):
                continue
            x = normalized[input_idx]
            y = targets[target_idx].T  # [scale, lead]
            mask = np.isfinite(y)
            if not mask.any():
                continue
            # Missing inputs invalidate the sample; missing targets are retained as masks.
            if not np.isfinite(x).all():
                continue
            windows[split]["X"].append(x.astype(np.float32))
            windows[split]["Y"].append(np.nan_to_num(y, nan=0.0).astype(np.float32))
            windows[split]["mask"].append(mask)
            last_spei = rows.iloc[input_idx[-1]][TARGETS].to_dict()
            windows[split]["metadata"].append(
                {
                    "dates": [str(d.date()) for d in dates[input_idx]],
                    "site": str(site),
                    "region": region,
                    "last_observed_spei": last_spei,
                    "spei_history": rows.iloc[input_idx][TARGETS]
                    .astype(float)
                    .values.tolist(),
                    "target_dates": [str(d.date()) for d in dates[target_idx]],
                    "index_source_dates": [
                        str((d - pd.DateOffset(months=1)).date())
                        for d in dates[input_idx]
                    ],
                    "index_source_dates_by_index": [
                        {
                            name: str((d - pd.DateOffset(months=lag)).date())
                            for name, lag in (
                                ("oni", 2), ("dmi", 1), ("nao", 1), ("pdo", 1)
                            )
                        }
                        for d in dates[input_idx]
                    ],
                }
            )
            if residual_heads:
                windows[split]["last_spei"].append(
                    np.nan_to_num(
                        np.asarray(
                            [last_spei[name] for name in TARGETS], dtype=np.float32
                        )
                    )
                )
            if static_context:
                windows[split]["static"].append(
                    np.asarray([latitude, aridity], dtype=np.float32)
                )
    split_summary = {
        name: {
            "n_windows": len(entries["metadata"]),
            "first_target": min((m["target_dates"][0] for m in entries["metadata"]), default=None),
            "last_target": max((m["target_dates"][-1] for m in entries["metadata"]), default=None),
        }
        for name, entries in windows.items()
    }
    if metadata_path:
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).write_text(
            json.dumps({"features": features, "normalizers": normalizers, "split_summary": split_summary, "split_rule": "all_inputs_and_targets_in_one_split"}, indent=2),
            encoding="utf-8",
        )
    result = {}
    for name, entries in windows.items():
        x = np.asarray(entries["X"], dtype=np.float32).reshape(
            -1, seq_len, len(features)
        )
        y = np.asarray(entries["Y"], dtype=np.float32).reshape(
            -1, len(TARGETS), forecast_steps
        )
        mask = np.asarray(entries["mask"], dtype=bool).reshape(
            -1, len(TARGETS), forecast_steps
        )
        last_spei = (
            torch.from_numpy(
                np.asarray(entries["last_spei"], dtype=np.float32).reshape(-1, 3)
            )
            if residual_heads
            else None
        )
        static = (
            torch.from_numpy(
                np.asarray(entries["static"], dtype=np.float32).reshape(-1, 2)
            )
            if static_context
            else None
        )
        result[name] = DroughtResearchDataset(
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(mask),
            entries["metadata"],
            last_spei=last_spei,
            static=static,
        )
    return result


def load_era_5_and_spei_data(
    inputs_csv_path: str,
    targets_nc_path: str,
    seq_len: int = 6,
    forecast_steps: int = 4,
):
    raise RuntimeError(
        "Legacy length-truncating NetCDF loader removed. Build data/monthly_sites.csv with data/build_dataset.py and use load_site_table()."
    )
