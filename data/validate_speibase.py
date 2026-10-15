"""Compare computed SPEI against official SPEIbase CSV or per-scale NetCDF files.

For NetCDF mode pass a directory containing spei03.nc, spei06.nc, spei12.nc.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from data.build_dataset import SITES


def compare(computed: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    keys = ["site", "date"]
    merged = computed[keys + ["spei_3", "spei_6", "spei_12"]].merge(
        reference[keys + ["spei_3", "spei_6", "spei_12"]], on=keys,
        suffixes=("_computed", "_reference"), validate="one_to_one")
    results = []
    for site, group in merged.groupby("site"):
        for scale in (3, 6, 12):
            a, b = group[f"spei_{scale}_computed"], group[f"spei_{scale}_reference"]
            valid = a.notna() & b.notna()
            if valid.sum() < 3:
                corr = rmse = np.nan
            else:
                corr = a[valid].corr(b[valid])
                rmse = float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))
            results.append({"site": site, "scale": scale, "n": int(valid.sum()), "correlation": corr, "rmse": rmse})
    return pd.DataFrame(results)


def _find_coord(ds, options):
    lookup = {name.lower(): name for name in list(ds.coords) + list(ds.variables)}
    for option in options:
        if option in lookup:
            return lookup[option]
    raise ValueError(f"Could not find coordinate matching {options}; available={list(ds.coords)}")


def extract_reference(netcdf_path: Path, variable: str, scale: int,
                     sites: dict | None = None) -> pd.DataFrame:
    ds = xr.open_dataset(netcdf_path)
    try:
        lat_name = _find_coord(ds, ("lat", "latitude"))
        lon_name = _find_coord(ds, ("lon", "longitude"))
        time_name = _find_coord(ds, ("time",))
        if variable not in ds:
            candidates = [name for name, item in ds.data_vars.items()
                          if {lat_name, lon_name, time_name}.issubset(set(item.dims))]
            if len(candidates) != 1:
                raise ValueError(f"Specify --variable; matching data variables are {candidates}")
            variable = candidates[0]
        records = []
        for site, location in (sites or SITES).items():
            point = ds[variable].sel({lat_name: location["lat"], lon_name: location["lon"]}, method="nearest")
            series = point.to_series()
            time_values = (series.index.get_level_values(time_name)
                           if isinstance(series.index, pd.MultiIndex) else series.index)
            dates = pd.to_datetime(time_values)
            values = np.asarray(series.values, dtype=float)
            fill = point.attrs.get("_FillValue", ds[variable].attrs.get("_FillValue"))
            if fill is not None:
                values[values == fill] = np.nan
            for date, value in zip(dates, values):
                records.append({"site": site, "date": pd.Timestamp(date).to_period("M").to_timestamp(),
                                f"spei_{scale}": value})
        return pd.DataFrame(records)
    finally:
        ds.close()


def compare_netcdf(computed: pd.DataFrame, directory: Path, variable: str | None = None,
                   sites: dict | None = None) -> pd.DataFrame:
    references = []
    for scale in (3, 6, 12):
        path = directory / f"spei{scale:02d}.nc"
        if not path.exists():
            raise FileNotFoundError(f"Expected SPEIbase file: {path}")
        references.append(extract_reference(path, variable, scale, sites))
    ref = references[0]
    for other in references[1:]:
        ref = ref.merge(other, on=["site", "date"], how="outer", validate="one_to_one")
    return compare(computed, ref)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("computed_csv")
    parser.add_argument("speibase", help="CSV with site/date/SPEI columns, or directory with spei03.nc/spei06.nc/spei12.nc")
    parser.add_argument("--variable", help="NetCDF SPEI variable name if it cannot be inferred")
    parser.add_argument("--config", default="configs/config.yaml", help="Site coordinate configuration")
    args = parser.parse_args()
    computed = pd.read_csv(args.computed_csv, parse_dates=["date"])
    reference_path = Path(args.speibase)
    if reference_path.is_dir():
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        result = compare_netcdf(computed, reference_path, args.variable, cfg.get("sites", SITES))
    else:
        result = compare(computed, pd.read_csv(reference_path, parse_dates=["date"]))
    print(result.to_string(index=False))
