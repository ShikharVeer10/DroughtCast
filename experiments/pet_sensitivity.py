"""Compare Hargreaves-Samani PET with TerraClimate reference ET at 12 sites."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import requests
import yaml

URL = "http://thredds.northwestknowledge.net:8080/thredds/ncss/grid/TERRACLIMATE_ALL/data/TerraClimate_pet_{year}.nc"


def _metrics(group: pd.DataFrame) -> dict:
    h = pd.to_numeric(group.pet_mm_hargreaves, errors="coerce").to_numpy()
    t = pd.to_numeric(group.pet_mm_terraclimate, errors="coerce").to_numpy()
    valid = np.isfinite(h) & np.isfinite(t)
    h, t = h[valid], t[valid]
    if len(h) < 3:
        return {"n": len(h), "correlation": None, "rmse_mm": None, "bias_mm": None}
    return {
        "n": len(h),
        "correlation": float(np.corrcoef(h, t)[0, 1]),
        "rmse_mm": float(np.sqrt(np.mean((h - t) ** 2))),
        "bias_mm": float(np.mean(h - t)),
    }


def retrieve(config_path: Path, dataset_path: Path, output_dir: Path, start_year: int, end_year: int) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sites = cfg["sites"]
    computed = pd.read_csv(dataset_path, parse_dates=["date"])
    computed = computed.loc[computed.date.between(f"{start_year}-01-01", f"{end_year}-12-01")]
    computed["site"] = computed.site.astype(str).str.lower()
    rows, retrievals = [], []
    for year in range(start_year, end_year + 1):
        params = {
            "var": "pet",
            "north": max(float(item["lat"]) for item in sites.values()) + 0.25,
            "south": min(float(item["lat"]) for item in sites.values()) - 0.25,
            "west": min(float(item["lon"]) for item in sites.values()) - 0.25,
            "east": max(float(item["lon"]) for item in sites.values()) + 0.25,
            "horizStride": "1",
            "time_start": f"{year}-01-01T00:00:00Z",
            "time_end": f"{year}-12-31T00:00:00Z",
            "accept": "netcdf",
        }
        response = requests.get(URL.format(year=year), params=params, timeout=120)
        response.raise_for_status()
        retrieved = datetime.now(UTC).isoformat()
        with netCDF4.Dataset("memory", memory=response.content) as ds:
            if "pet" not in ds.variables or "lat" not in ds.variables or "lon" not in ds.variables:
                raise ValueError(f"Unexpected TerraClimate subset variables in {year}")
            variable = ds.variables["pet"]
            units = str(getattr(variable, "units", ""))
            if units.lower() != "mm":
                raise ValueError(f"TerraClimate pet units expected mm/month, got {units!r}")
            times = pd.to_datetime([
                str(item)[:10]
                for item in netCDF4.num2date(
                    ds.variables["time"][:], ds.variables["time"].units,
                    calendar=getattr(ds.variables["time"], "calendar", "standard"),
                    only_use_cftime_datetimes=False,
                )
            ]).to_period("M").to_timestamp()
            latitudes = np.asarray(ds.variables["lat"][:], dtype=float)
            longitudes = np.asarray(ds.variables["lon"][:], dtype=float)
            data = variable[:]
            if not np.ma.isMaskedArray(data):
                data = np.ma.masked_invalid(data)
            for site, spec in sites.items():
                iy = int(np.abs(latitudes - float(spec["lat"])).argmin())
                ix = int(np.abs(longitudes - float(spec["lon"])).argmin())
                values = np.asarray(data[:, iy, ix].filled(np.nan), dtype=float)
                rows.extend({
                    "site": site,
                    "region": spec["region"],
                    "date": date,
                    "pet_mm_terraclimate": value,
                    "terra_lat": float(latitudes[iy]),
                    "terra_lon": float(longitudes[ix]),
                } for date, value in zip(times, values))
        retrievals.append({
            "year": year,
            "request_url": response.url,
            "retrieved_utc": retrieved,
            "status": response.status_code,
            "content_length": len(response.content),
        })
    terra = pd.DataFrame(rows)
    merged = computed[["site", "date", "region", "pet_mm"]].rename(
        columns={"pet_mm": "pet_mm_hargreaves"}
    ).merge(terra, on=["site", "date", "region"], how="inner", validate="one_to_one")
    expected = (end_year - start_year + 1) * 12 * len(sites)
    if len(merged) != expected or merged.pet_mm_terraclimate.isna().any():
        raise ValueError(f"Incomplete PET comparison coverage: {len(merged)} of {expected} site-months")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "pet_matched_monthly.csv", index=False)
    rows_metrics = [{"group_type": "site", "group": site, **_metrics(group)} for site, group in merged.groupby("site", sort=True)]
    rows_metrics.extend({"group_type": "region", "group": region, **_metrics(group)} for region, group in merged.groupby("region", sort=True))
    rows_metrics.append({"group_type": "all", "group": "Maharashtra", **_metrics(merged)})
    metrics = pd.DataFrame(rows_metrics)
    metrics.to_csv(output_dir / "pet_comparison.csv", index=False)
    metadata = {
        "reference_product": "TerraClimate v1.1 monthly reference evapotranspiration (pet)",
        "source_url_template": URL,
        "license": "CC0 1.0",
        "reference_units": "mm/month (monthly totals)",
        "calculated_pet_method": "monthly Hargreaves-Samani from NASA POWER monthly precipitation-independent temperature aggregates",
        "spatial_match": "nearest TerraClimate 1/24-degree grid cell for every configured Maharashtra site",
        "period": [f"{start_year}-01", f"{end_year}-12"],
        "coverage_site_months": len(merged),
        "retrievals": retrievals,
        "files": {"matched_monthly": "pet_matched_monthly.csv", "metrics": "pet_comparison.csv"},
        "interpretation": "Product-to-product sensitivity check, not station validation; TerraClimate PET is an independent modeled reference ET product.",
    }
    (output_dir / "pet_comparison_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", default="results/maharashtra/protocol7_final_local/monthly_sites.csv")
    parser.add_argument("--output-dir", default="results/maharashtra/pet_sensitivity")
    parser.add_argument("--start-year", type=int, default=1990)
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()
    retrieve(Path(args.config), Path(args.dataset), Path(args.output_dir), args.start_year, args.end_year)


if __name__ == "__main__":
    main()
