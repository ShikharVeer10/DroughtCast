"""Compare Maharashtra computed SPEI targets with official SPEIbase v2.11."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml


def _coord_name(dataset: xr.Dataset, options: tuple[str, ...]) -> str:
    names = {name.lower(): name for name in (*dataset.coords, *dataset.variables)}
    for option in options:
        if option in names:
            return names[option]
    raise ValueError(f"Missing coordinate {options}; found {list(dataset.coords)}")


def extract_reference(path: Path, scale: int, sites: dict) -> pd.DataFrame:
    rows = []
    with xr.open_dataset(path) as ds:
        lat_name = _coord_name(ds, ("lat", "latitude"))
        lon_name = _coord_name(ds, ("lon", "longitude"))
        time_name = _coord_name(ds, ("time",))
        candidates = [
            name for name, var in ds.data_vars.items()
            if {lat_name, lon_name, time_name}.issubset(set(var.dims))
        ]
        var_name = "spei" if "spei" in ds.data_vars else (candidates[0] if len(candidates) == 1 else None)
        if var_name is None:
            raise ValueError(f"Cannot identify SPEI variable in {path}: {candidates}")
        site_names = list(sites)
        site_coord = xr.DataArray(site_names, dims="site", name="site")
        lat_values = xr.DataArray([float(sites[s]["lat"]) for s in site_names], dims="site", coords={"site": site_coord})
        lon_values = xr.DataArray([float(sites[s]["lon"]) for s in site_names], dims="site", coords={"site": site_coord})
        points = ds[var_name].sel({lat_name: lat_values, lon_name: lon_values}, method="nearest").transpose(time_name, "site")
        values = np.asarray(points.values, dtype=float)
        fill = points.attrs.get("_FillValue", ds[var_name].attrs.get("_FillValue", ds[var_name].encoding.get("_FillValue")))
        if fill is not None:
            values[values == float(fill)] = np.nan
        values[~np.isfinite(values) | (np.abs(values) > 1e10)] = np.nan
        dates = pd.to_datetime(ds[time_name].values).to_period("M").to_timestamp()
        nearest_lats = np.asarray(ds[lat_name].sel({lat_name: lat_values}, method="nearest").values, dtype=float)
        nearest_lons = np.asarray(ds[lon_name].sel({lon_name: lon_values}, method="nearest").values, dtype=float)
        for column, site in enumerate(site_names):
            rows.extend({
                "site": site,
                "date": date,
                f"spei_{scale}_reference": value,
                "reference_lat": nearest_lats[column],
                "reference_lon": nearest_lons[column],
            } for date, value in zip(dates, values[:, column]))
    return pd.DataFrame(rows)


def _metrics(group: pd.DataFrame, scale: int) -> dict:
    a = group[f"spei_{scale}_computed"].to_numpy(dtype=float)
    b = group[f"spei_{scale}_reference"].to_numpy(dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    result = {"n": int(valid.sum())}
    if len(a) < 3:
        return result | {"correlation": np.nan, "rmse": np.nan, "bias": np.nan}
    result.update({
        "correlation": float(np.corrcoef(a, b)[0, 1]),
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "bias": float(np.mean(a - b)),
    })
    for label, threshold in (("drought", -1.0), ("severe", -1.5), ("extreme", -2.0)):
        comp, ref = a <= threshold, b <= threshold
        result[f"{label}_agreement"] = float(np.mean(comp == ref))
        result[f"{label}_computed_events"] = int(comp.sum())
        result[f"{label}_reference_events"] = int(ref.sum())
        result[f"{label}_both_events"] = int((comp & ref).sum())
        precision = float((comp & ref).sum() / comp.sum()) if comp.sum() else np.nan
        recall = float((comp & ref).sum() / ref.sum()) if ref.sum() else np.nan
        result[f"{label}_precision"] = precision
        result[f"{label}_recall"] = recall
        result[f"{label}_f1"] = (
            float(2 * precision * recall / (precision + recall))
            if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
            else np.nan
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", default="data/maharashtra/monthly_sites.csv")
    parser.add_argument("--reference-dir", default="data/speibase_v2_11")
    parser.add_argument("--output-dir", default="results/maharashtra/speibase_validation")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sites = cfg["sites"]
    expected = set(cfg["data"]["site_filter"])
    if set(sites) != expected or len(sites) != 12:
        raise ValueError("Config must define exactly the 12 selected Maharashtra sites")
    computed = pd.read_csv(args.dataset, parse_dates=["date"])
    computed["site"] = computed["site"].astype(str).str.lower().str.strip()
    computed = computed[computed["site"].isin(expected)].copy()
    if set(computed["site"].unique()) != expected:
        raise ValueError("Computed target table does not contain all configured Maharashtra sites")

    references = []
    for scale in (3, 6, 12):
        path = Path(args.reference_dir) / f"spei{scale:02d}.nc"
        if not path.is_file():
            raise FileNotFoundError(f"Official SPEIbase file not found: {path}")
        references.append(extract_reference(path, scale, sites))
    reference = references[0]
    for item in references[1:]:
        reference = reference.merge(
            item.drop(columns=["reference_lat", "reference_lon"]),
            on=["site", "date"], how="outer", validate="one_to_one",
        )
    merged = computed[["site", "date", "spei_3", "spei_6", "spei_12"]].merge(
        reference, on=["site", "date"], how="inner", validate="one_to_one",
        suffixes=("_computed", "_reference"),
    )
    merged = merged.rename(columns={f"spei_{s}": f"spei_{s}_computed" for s in (3, 6, 12)})

    metric_rows = []
    for site, group in merged.groupby("site", sort=True):
        for scale in (3, 6, 12):
            metric_rows.append({"site": site, "scale": scale, **_metrics(group, scale)})
    metrics = pd.DataFrame(metric_rows)
    overall_rows = []
    for scale in (3, 6, 12):
        overall_rows.append({"site": "ALL", "scale": scale, **_metrics(merged, scale)})
    metrics = pd.concat([metrics, pd.DataFrame(overall_rows)], ignore_index=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "site_scale_metrics.csv", index=False)
    merged.to_csv(output_dir / "matched_monthly_points.csv", index=False)
    metadata = {
        "reference": "SPEIbase v2.11",
        "source_url": "https://spei.csic.es/spei_database_2_11/nc/",
        "reference_resolution_degrees": 0.5,
        "extraction": "nearest reference grid cell to each configured site; exact same monthly dates",
        "comparison_period": [str(merged.date.min().date()), str(merged.date.max().date())],
        "sites": sorted(expected),
        "site_reference_cells": merged[["site", "reference_lat", "reference_lon"]].drop_duplicates().sort_values("site").to_dict("records"),
        "computed_method": "NASA POWER monthly precipitation and Hargreaves-Samani PET; nonparametric SPEI, 1990-2010 calibration",
        "files": {"metrics": str(output_dir / "site_scale_metrics.csv"), "matched_points": str(output_dir / "matched_monthly_points.csv")},
    }
    (output_dir / "validation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics[metrics.site == "ALL"].to_string(index=False))
    print(f"Saved detailed validation to {output_dir}")


if __name__ == "__main__":
    main()
