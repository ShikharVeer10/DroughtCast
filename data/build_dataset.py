"""Build date-aligned monthly site data from NASA POWER and NOAA PSL.

PET uses a monthly Hargreaves-Samani approximation from POWER T2M/T2M_MAX/
T2M_MIN and extraterrestrial radiation. This is not FAO-56 Penman-Monteith.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import requests

from data.protocol import DATASET_PROTOCOL_VERSION, INDEX_ALIGNMENT
from data.quality import target_quality_report, validate_target_quality

POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
POWER_VARS = [
    "PRECTOTCORR",
    "T2M",
    "T2MDEW",
    "WS10M",
    "ALLSKY_SFC_SW_DWN",
    "T2M_MAX",
    "T2M_MIN",
]
NOAA = {
    "oni": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    "dmi": "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data",
    "nao": "https://psl.noaa.gov/data/correlation/nao.data",
    "pdo": "https://psl.noaa.gov/data/correlation/pdo.data",
}
SITES = {
    "guntur": {"lat": 15.9129, "lon": 79.74, "region": "south_asia"},
    "jaipur": {"lat": 26.9124, "lon": 75.7873, "region": "south_asia"},
    "nagpur": {"lat": 21.1458, "lon": 79.0882, "region": "south_asia"},
    "murray_darling": {"lat": -34.0, "lon": 143.0, "region": "australia"},
    "addis_ababa": {"lat": 9.03, "lon": 38.74, "region": "east_africa"},
    "phoenix": {"lat": 33.4484, "lon": -112.074, "region": "us_southwest"},
    "seville": {"lat": 37.3891, "lon": -5.9845, "region": "mediterranean"},
    "pretoria": {"lat": -25.7479, "lon": 28.2293, "region": "southern_africa"},
}


def fetch_power(
    site: str, spec: dict[str, Any], start: int, end: int, cache: Path
) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{site}_power_{start}_{end}.csv"
    if path.exists():
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.set_index("date")
    response = requests.get(
        POWER_URL,
        params={
            "parameters": ",".join(POWER_VARS),
            "community": "AG",
            "longitude": spec["lon"],
            "latitude": spec["lat"],
            "start": start,
            "end": end,
            "format": "JSON",
        },
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()
    try:
        parameters = body["properties"]["parameter"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Unexpected NASA POWER response for {site}: {body!r}"
        ) from exc
    if not all(name in parameters for name in POWER_VARS):
        raise ValueError(
            f"NASA POWER response missing expected variables: {list(parameters)}"
        )
    frame = pd.DataFrame(parameters)
    # Monthly POWER includes annual summaries labelled YYYY13; reject before date parsing.
    frame = frame.loc[[str(key)[-2:] != "13" for key in frame.index]]
    frame.index = frame.index.map(str)
    frame.index = pd.to_datetime(frame.index, format="%Y%m", errors="raise")
    frame.index.name = "date"
    frame = frame.sort_index()
    tmp_path = path.with_suffix(".tmp")
    frame.to_csv(tmp_path, index_label="date")
    tmp_path.replace(path)
    return frame


def _parse_noaa_text(name: str, text: str) -> pd.Series:
    rows: dict[pd.Timestamp, float] = {}
    if name == "oni":
        season_month = {
            "DJF": 1,
            "JFM": 2,
            "FMA": 3,
            "MAM": 4,
            "AMJ": 5,
            "MJJ": 6,
            "JJA": 7,
            "JAS": 8,
            "ASO": 9,
            "SON": 10,
            "OND": 11,
            "NDJ": 12,
        }
        for line in text.splitlines():
            match = re.match(
                r"\s*(DJF|JFM|FMA|MAM|AMJ|MJJ|JJA|JAS|ASO|SON|OND|NDJ)\s+(\d{4})\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)",
                line,
            )
            if match:
                season, year, _sst, anomaly = match.groups()
                rows[pd.Timestamp(int(year), season_month[season], 1)] = float(anomaly)
    else:
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 13 or not parts[0].isdigit() or len(parts[0]) != 4:
                continue
            year = int(parts[0])
            try:
                # PSL annual tables sometimes append a 13th annual mean after January–December.
                values = [float(v) for v in parts[1:13]]
            except ValueError:
                continue
            for month, val in enumerate(values, 1):
                if np.isfinite(val) and val > -90:
                    rows[pd.Timestamp(year, month, 1)] = val
    if not rows:
        raise ValueError(
            f"Could not parse NOAA {name} data; source format may have changed"
        )
    return pd.Series(rows, name=name).sort_index()


def fetch_indices(start: str, end: str, cache: Path) -> pd.DataFrame:
    columns = []
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in NOAA.items():
        path = cache / f"noaa_{name}.txt"
        if path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"Could not fetch NOAA {name} from {url}. Check outbound HTTPS access and retry."
                ) from exc
            text = response.text
            path.write_text(text, encoding="utf-8")
        columns.append(_parse_noaa_text(name, text))
    index = pd.concat(columns, axis=1, sort=True).sort_index()
    return index.loc[start:end]


def mid_month_doy(index: pd.DatetimeIndex) -> np.ndarray:
    """Day-of-year at mid-month (not the 1st) for monthly Hargreaves Ra."""
    days = index.days_in_month.to_numpy(dtype=float)
    mid = index + pd.to_timedelta((days - 1.0) / 2.0, unit="D")
    return mid.day_of_year.to_numpy(dtype=float)


def estimate_pet_hargreaves(frame: pd.DataFrame, latitude: float) -> pd.Series:
    """Monthly Hargreaves-Samani PET (mm/month) from POWER monthly extrema.

    ET0 (mm/day) = 0.0023 * (Tmean + 17.8) * sqrt(TR) * (Ra / 2.45),
    where Ra is extraterrestrial radiation in MJ/m²/day computed at mid-month.
    """
    tmean = frame["T2M"].astype(float)
    # POWER monthly T2M_MAX/MIN are monthly means of daily extrema, not individual daily series.
    # Hargreaves-Samani is therefore an approximation at monthly resolution.
    trange = (frame["T2M_MAX"].astype(float) - frame["T2M_MIN"].astype(float)).clip(
        lower=0
    )
    doy = mid_month_doy(frame.index)
    phi = np.deg2rad(latitude)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    ra_extraterrestrial = (
        (24 * 60 / np.pi)
        * 0.0820
        * dr
        * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    )
    et0_day = (
        0.0023
        * (tmean + 17.8)
        * np.sqrt(trange)
        * (np.maximum(ra_extraterrestrial, 0) / 2.45)
    )
    return pd.Series(
        np.maximum(et0_day, 0) * frame.index.days_in_month,
        index=frame.index,
        name="pet_mm",
    )


def _tail_slopes(
    sorted_cal: np.ndarray, z_cal: np.ndarray, k: int = 5
) -> tuple[float, float]:
    """Stable dry/wet tail slopes (z per raw unit) for extrapolation."""
    edge_lo = (z_cal[1] - z_cal[0]) / (sorted_cal[1] - sorted_cal[0])
    edge_hi = (z_cal[-1] - z_cal[-2]) / (sorted_cal[-1] - sorted_cal[-2])
    if len(sorted_cal) < k + 1:
        return float(edge_lo), float(edge_hi)
    lo = np.polyfit(sorted_cal[: k + 1], z_cal[: k + 1], 1)[0]
    hi = np.polyfit(sorted_cal[-(k + 1) :], z_cal[-(k + 1) :], 1)[0]
    cap = 3.0 * np.median(np.diff(z_cal) / np.diff(sorted_cal))
    return float(min(max(lo, 0.0), cap)), float(min(max(hi, 0.0), cap))


def _nonparametric_z_scores(
    values: np.ndarray, calibration: np.ndarray, max_abs_z: float = 3.5
) -> np.ndarray:
    """Map values to standard normals via Gringorten plotting positions.

    Uses one consistent plotting position for in-sample and out-of-sample
    values, linear interpolation of z between calibration quantiles, and
    linear extrapolation beyond the calibration min/max. Tails use the
    least-squares slope of the outer 5 calibration points, capped at 3x the
    median interior slope; |z| is clipped to max_abs_z (default 3.5, to be
    validated against SPEIbase).
    """
    if max_abs_z <= 0:
        raise ValueError("max_abs_z must be positive")
    calibration = np.asarray(calibration, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    if np.isnan(calibration).any():
        return result
    sorted_cal = np.sort(calibration)
    sorted_cal = sorted_cal[np.isfinite(sorted_cal)]
    if len(sorted_cal) > 1:
        # Duplicate calibration values make tail slopes undefined. Collapse
        # them before fitting the plotting-position transform.
        sorted_cal = np.unique(sorted_cal)
    n = len(sorted_cal)
    if n == 1:
        result[np.isfinite(values)] = 0.0
        return np.clip(result, -max_abs_z, max_abs_z)
    if n < 1:
        raise ValueError("nonparametric SPEI needs at least 2 calibration samples")
    ranks = np.arange(1, n + 1, dtype=float)
    p_cal = (ranks - 0.44) / (n + 0.12)
    inv = NormalDist().inv_cdf
    z_cal = np.asarray([inv(float(p)) for p in p_cal], dtype=float)
    slope_lo, slope_hi = _tail_slopes(sorted_cal, z_cal)
    for i, raw in enumerate(values):
        v = float(raw)
        if not np.isfinite(v):
            continue
        if v <= sorted_cal[0]:
            result[i] = z_cal[0] + slope_lo * (v - sorted_cal[0])
        elif v >= sorted_cal[-1]:
            result[i] = z_cal[-1] + slope_hi * (v - sorted_cal[-1])
        else:
            result[i] = float(np.interp(v, sorted_cal, z_cal))
    return np.clip(result, -max_abs_z, max_abs_z)


def _standardize_accumulation(
    balance: pd.Series, scale: int, method: str, calibration_end: str = "2010-12-01"
) -> pd.Series:
    accum = balance.rolling(scale, min_periods=scale).sum()
    result = pd.Series(np.nan, index=balance.index, dtype=float)
    calibration_end = pd.Timestamp(calibration_end)
    for month in range(1, 13):
        subset = accum.loc[
            (accum.index.month == month) & (accum.index <= calibration_end)
        ].dropna()
        values = accum.loc[accum.index.month == month]
        if len(subset) < 20:
            continue
        if method == "nonparametric":
            z = _nonparametric_z_scores(
                values.to_numpy(dtype=float), subset.to_numpy(dtype=float)
            )
            result.loc[values.index] = z
        else:
            # Fit a three-parameter log-logistic (Fisk) by moments after shifting positive.
            cal_values = subset.to_numpy(dtype=float)
            loc = cal_values.min() - max(cal_values.std() * 1e-3, 1e-6)
            logs = np.log(cal_values - loc)
            shape = max(1.0, np.pi / (np.sqrt(3) * max(logs.std(ddof=1), 1e-6)))
            scale_param = np.exp(logs.mean())
            all_values = accum.loc[accum.index.month == month]
            shifted = np.maximum(all_values.to_numpy() - loc, 1e-12)
            cdf = 1.0 / (1.0 + (shifted / scale_param) ** (-shape))
            result.loc[all_values.index] = pd.Series(
                np.clip(cdf, 1e-6, 1 - 1e-6), index=all_values.index
            ).map(NormalDist().inv_cdf)
    return result


def build_site(
    site: str,
    spec: dict[str, Any],
    start: int,
    end: int,
    cache: Path,
    method: str,
    indices: pd.DataFrame,
) -> pd.DataFrame:
    frame = fetch_power(site, spec, start, end, cache)
    frame = frame.loc[frame.index.month != 13]
    expected = pd.date_range(f"{start}-01-01", f"{end}-12-01", freq="MS")
    frame = frame.reindex(expected)
    frame.index.name = "date"
    frame["pet_mm"] = estimate_pet_hargreaves(frame, spec["lat"])
    # NASA POWER PRECTOTCORR is mm/day; convert to a monthly total.
    balance = frame["PRECTOTCORR"] * frame.index.days_in_month - frame["pet_mm"]
    for scale in (3, 6, 12):
        frame[f"spei_{scale}"] = _standardize_accumulation(balance, scale, method)
    for variable in POWER_VARS:
        frame[f"lag_{variable}"] = frame[variable].shift(1)
    # Observed SPEI at the input month (unshifted) for information parity with persistence.
    for scale in (3, 6, 12):
        frame[f"spei_{scale}_obs"] = frame[f"spei_{scale}"]
    frame = frame.join(indices, how="left")
    # For each monthly index, attach the previous month's published value to
    # month m. This is conservative for centered ONI (which includes m+1)
    # and gives a single, auditable alignment rule across all indices.
    for name in ("oni", "dmi", "nao", "pdo"):
        if name in frame.columns:
            frame[name] = frame[name].shift(1)
    frame["site"] = site
    frame["region"] = spec["region"]
    return frame.reset_index()


def _resolve_data_variable(
    dataset: Any, requested: str | None, lat_name: str, lon_name: str, time_name: str
) -> str:
    if requested:
        if requested not in dataset.data_vars:
            raise ValueError(
                f"SPEIbase variable {requested!r} not found; variables={list(dataset.data_vars)}"
            )
        return requested
    candidates = [
        name
        for name, var in dataset.data_vars.items()
        if {lat_name, lon_name, time_name}.issubset(set(var.dims))
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Could not infer SPEIbase data variable; candidates={candidates}. Pass --speibase-variable."
        )
    return candidates[0]


def ingest_speibase(
    computed: pd.DataFrame,
    directory: Path,
    sites: dict[str, Any],
    variable: str | None = None,
) -> pd.DataFrame:
    """Replace computed targets with nearest-cell official SPEIbase values."""
    import xarray as xr

    reference = None
    for scale in (3, 6, 12):
        path = directory / f"spei{scale:02d}.nc"
        if not path.is_file():
            raise FileNotFoundError(f"Expected official SPEIbase file {path}")
        with xr.open_dataset(path) as ds:
            names = {
                name.lower(): name for name in list(ds.coords) + list(ds.variables)
            }

            def find(*choices, names=names, path=path, coords=ds.coords):
                for choice in choices:
                    if choice in names:
                        return names[choice]
                raise ValueError(
                    f"No coordinate {choices} in {path}; found {list(coords)}"
                )

            lat_name, lon_name, time_name = (
                find("lat", "latitude"),
                find("lon", "longitude"),
                find("time"),
            )
            var_name = _resolve_data_variable(
                ds, variable, lat_name, lon_name, time_name
            )
            rows = []
            for site, spec in sites.items():
                point = ds[var_name].sel(
                    {lat_name: spec["lat"], lon_name: spec["lon"]}, method="nearest"
                )
                series = point.to_series()
                time_values = (
                    series.index.get_level_values(time_name)
                    if isinstance(series.index, pd.MultiIndex)
                    else series.index
                )
                time_index = pd.to_datetime(time_values)
                values = np.asarray(series.values, dtype=float)
                fill = point.attrs.get(
                    "_FillValue", ds[var_name].attrs.get("_FillValue")
                )
                if fill is not None:
                    values[values == fill] = np.nan
                rows.extend(
                    {
                        "site": site,
                        "date": pd.Timestamp(d)
                        .tz_localize(None)
                        .to_period("M")
                        .to_timestamp(),
                        f"spei_{scale}": v,
                    }
                    for d, v in zip(time_index, values)
                )
            scale_frame = pd.DataFrame(rows)
            reference = (
                scale_frame
                if reference is None
                else reference.merge(
                    scale_frame, on=["site", "date"], how="outer", validate="one_to_one"
                )
            )
    target_cols = [f"spei_{scale}" for scale in (3, 6, 12)]
    left = computed.drop(
        columns=[c for c in target_cols if c in computed], errors="ignore"
    )
    result = left.merge(
        reference, on=["site", "date"], how="left", validate="one_to_one"
    )
    if not result[target_cols].notna().any(axis=0).all():
        missing_scales = [
            scale for scale in target_cols if not result[scale].notna().any()
        ]
        raise ValueError(
            f"SPEIbase extraction produced no valid values for {missing_scales}"
        )
    # Keep observed state features consistent with the official target series.
    for scale in (3, 6, 12):
        observed = f"spei_{scale}_obs"
        if observed in result:
            result[observed] = result[f"spei_{scale}"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="data/monthly_sites.csv")
    parser.add_argument("--cache", default="data/raw")
    parser.add_argument("--start", type=int, default=1981)
    parser.add_argument("--end", type=int, default=2024)
    parser.add_argument(
        "--spei-method",
        choices=("nonparametric", "log-logistic"),
        default="nonparametric",
    )
    parser.add_argument(
        "--speibase",
        type=Path,
        help="Optional local official SPEIbase NetCDF directory",
    )
    parser.add_argument(
        "--speibase-variable", help="Optional SPEIbase NetCDF variable name"
    )
    args = parser.parse_args()
    cfg = (
        json.loads(Path(args.config).read_text(encoding="utf-8"))
        if args.config.endswith(".json")
        else None
    )
    if cfg is None:
        import yaml

        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sites = cfg.get("sites", SITES)
    indices = fetch_indices(f"{args.start}-01", f"{args.end}-12", Path(args.cache))
    data = pd.concat(
        [
            build_site(
                name,
                spec,
                args.start,
                args.end,
                Path(args.cache),
                args.spei_method,
                indices,
            )
            for name, spec in sites.items()
        ],
        ignore_index=True,
    )
    if args.speibase:
        from data.validate_speibase import compare_netcdf, speibase_points

        validation = compare_netcdf(data, args.speibase, args.speibase_variable, sites)
        validation_path = Path("results/speibase_validation.csv")
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        validation.to_csv(validation_path, index=False)
        points = data[["site", "date", "spei_3", "spei_6", "spei_12"]].merge(
            speibase_points(args.speibase, args.speibase_variable, sites),
            on=["site", "date"],
            how="inner",
            suffixes=("_computed", "_speibase"),
        )
        point_rows = []
        for scale in (3, 6, 12):
            point_rows.append(
                points[
                    ["site", "date", f"spei_{scale}_computed", f"spei_{scale}_speibase"]
                ]
                .rename(
                    columns={
                        f"spei_{scale}_computed": "computed_spei",
                        f"spei_{scale}_speibase": "speibase_spei",
                    }
                )
                .assign(scale=scale)
            )
        pd.concat(point_rows, ignore_index=True).to_csv(
            Path("results/speibase_validation_points.csv"), index=False
        )
        data = ingest_speibase(data, args.speibase, sites, args.speibase_variable)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    quality = target_quality_report(data)
    validate_target_quality(quality)
    (output.parent / "target_quality.json").write_text(
        json.dumps(quality, indent=2, allow_nan=True), encoding="utf-8"
    )
    (output.parent / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": DATASET_PROTOCOL_VERSION,
                "source": "NASA POWER monthly point; NOAA PSL teleconnection indices",
                "period": [args.start, args.end],
                "spei_method": args.spei_method,
                "spei_calibration_period": [args.start, "2010-12-01"],
                "pet_method": "Monthly Hargreaves-Samani approximation using POWER T2M, T2M_MAX, and T2M_MIN monthly aggregates",
                "sites": sites,
                "climate_variables": POWER_VARS,
                "teleconnection_indices": list(NOAA),
                "model_features": POWER_VARS[:5]
                + list(NOAA)
                + ["spei_3_obs", "spei_6_obs", "spei_12_obs"],
                "index_alignment": INDEX_ALIGNMENT,
                "speibase": str(args.speibase) if args.speibase else None,
                "target_quality_report": str(output.parent / "target_quality.json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(data)} site-month rows to {output}")


if __name__ == "__main__":
    main()
