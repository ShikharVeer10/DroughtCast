from __future__ import annotations

import argparse
import hashlib
import json
import re
import warnings
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import requests

from data.protocol import (
    DATASET_PROTOCOL_VERSION,
    INDEX_ALIGNMENT,
    SPEI_CALIBRATION_END,
    SPLIT_BOUNDARIES,
)
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
SITES: dict[str, dict[str, Any]] = {}

ALLOWED_MAHARASHTRA_REGIONS = {
    "marathwada", "vidarbha", "madhya_maharashtra", "konkan"
}
EXPECTED_MAHARASHTRA_SITES = {
    "aurangabad", "latur", "beed", "nanded", "nagpur", "akola",
    "amravati", "pune", "solapur", "nashik", "mumbai", "ratnagiri",
}


def _normalize_sites(sites: Any) -> dict[str, dict[str, Any]]:
    """Normalize explicitly configured Maharashtra sites; never use global sites."""
    if sites is None:
        return {}
    if isinstance(sites, list):
        sites = {
            s.get("id", s.get("name", "").lower().replace(" ", "_")): s
            for s in sites
        }
    if not isinstance(sites, dict):
        raise ValueError("Configured sites must be a mapping or list")
    return {str(name).strip().lower(): spec for name, spec in sites.items()}


def _validate_maharashtra_sites(sites: dict[str, dict[str, Any]]) -> None:
    if set(sites) != EXPECTED_MAHARASHTRA_SITES:
        raise ValueError(
            "Configuration must contain exactly the 12 Maharashtra sites; "
            f"missing={sorted(EXPECTED_MAHARASHTRA_SITES-set(sites))}, "
            f"unexpected={sorted(set(sites)-EXPECTED_MAHARASHTRA_SITES)}"
        )
    for site, spec in sites.items():
        region = str(spec.get("region", "")).strip().lower().replace(" ", "_")
        if region not in ALLOWED_MAHARASHTRA_REGIONS:
            raise ValueError(f"Invalid Maharashtra region for {site}: {spec.get('region')!r}")
        if not np.isfinite([spec.get("lat", np.nan), spec.get("lon", np.nan)]).all():
            raise ValueError(f"Missing finite coordinates for Maharashtra site {site}")


def fetch_power(
    site: str, spec: dict[str, Any], start: int, end: int, cache: Path,
    end_month: int = 12, refresh: bool = False,
) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{site}_power_{start}_{end}.csv"
    expected = pd.date_range(f"{start}-01-01", f"{end}-{end_month:02d}-01", freq="MS")
    if path.exists() and not refresh:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        cached = cached.apply(pd.to_numeric, errors="coerce").replace(
            [-999.0, -99.0, -9999.0, -32768.0], np.nan
        )
        if set(POWER_VARS).issubset(cached.columns) and expected.isin(cached.index).all():
            cached = cached.loc[expected[0]:expected[-1]].reindex(expected)
            sidecar = path.with_suffix(".metadata.json")
            if not sidecar.exists():
                sidecar.write_text(json.dumps({
                    "units": {}, "unit_validation": "not verified for legacy cache",
                    "retrieved_utc": None,
                }, indent=2), encoding="utf-8")
            return cached
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
    unit_table = body.get("parameters", {})
    expected_units = {
        "PRECTOTCORR": {"mm/day"},
        "T2M": {"c"}, "T2MDEW": {"c"}, "T2M_MAX": {"c"}, "T2M_MIN": {"c"},
        "WS10M": {"m/s"},
        "ALLSKY_SFC_SW_DWN": {"mj/m^2/day", "kwh/m^2/day", "kw-hr/m^2/day"},
    }
    for variable, accepted in expected_units.items():
        reported = unit_table.get(variable, {})
        unit = str(reported.get("units", "") if isinstance(reported, dict) else reported)
        unit = unit.lower().replace(" ", "")
        if unit not in accepted:
            raise ValueError(
                f"NASA POWER {variable} unit {unit!r} is not one of {sorted(accepted)}"
            )
    frame = pd.DataFrame(parameters)
    # Monthly POWER includes annual summaries labelled YYYY13; reject before date parsing.
    frame = frame.loc[[str(key)[-2:] != "13" for key in frame.index]]
    frame.index = frame.index.map(str)
    frame.index = pd.to_datetime(frame.index, format="%Y%m", errors="raise")
    frame.index.name = "date"
    frame = frame.sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce").replace(
        [-999.0, -99.0, -9999.0, -32768.0], np.nan
    )
    missing = expected.difference(frame.index)
    if len(missing):
        raise ValueError(
            f"NASA POWER returned incomplete monthly dates for {site}: "
            f"{missing.strftime('%Y-%m').tolist()}"
        )
    tmp_path = path.with_suffix(".tmp")
    frame.to_csv(tmp_path, index_label="date")
    tmp_path.replace(path)
    path.with_suffix(".metadata.json").write_text(
        json.dumps({"units": {k: v.get("units") for k, v in unit_table.items()},
                    "header": body.get("header", {}), "unit_validation": "passed",
                    "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat()}, indent=2),
        encoding="utf-8",
    )
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
                # NOAA PSL series use both -99.9 and -9.9 sentinels.
                if np.isfinite(val) and val > -90 and not np.isclose(val, -9.9):
                    rows[pd.Timestamp(year, month, 1)] = val
    if not rows:
        raise ValueError(
            f"Could not parse NOAA {name} data; source format may have changed"
        )
    return pd.Series(rows, name=name).sort_index()


def fetch_indices(start: str, end: str, cache: Path, refresh: bool = False) -> pd.DataFrame:
    columns = []
    provenance = {}
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in NOAA.items():
        path = cache / f"noaa_{name}.txt"
        if path.exists() and not refresh:
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
            path.with_suffix(".metadata.json").write_text(
                json.dumps({
                    "source_url": url,
                    "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }, indent=2), encoding="utf-8",
            )
        sidecar = path.with_suffix(".metadata.json")
        if not sidecar.is_file():
            sidecar.write_text(json.dumps({
                "source_url": url,
                "retrieval_date": None,
                "unit_validation": "not applicable for index; legacy cache retrieval is unknown",
            }, indent=2), encoding="utf-8")
        provenance[name] = json.loads(sidecar.read_text(encoding="utf-8"))
        columns.append(_parse_noaa_text(name, text))
    index = pd.concat(columns, axis=1, sort=True).sort_index()
    result = index.loc[start:end]
    result.attrs["provenance"] = provenance
    return result


def mid_month_doy(index: pd.DatetimeIndex) -> np.ndarray:
    """Day-of-year at mid-month (not the 1st) for monthly Hargreaves Ra."""
    days = index.days_in_month.to_numpy(dtype=float)
    mid = index + pd.to_timedelta((days - 1.0) / 2.0, unit="D")
    return mid.day_of_year.to_numpy(dtype=float)


def estimate_pet_hargreaves(frame: pd.DataFrame, latitude: float) -> pd.Series:
    tmean = frame["T2M"].astype(float)
    # POWER monthly T2M_MAX/MIN are monthly means of daily extrema, not individual daily series.
    # Hargreaves-Samani is therefore an approximation at monthly resolution.
    trange = (frame["T2M_MAX"].astype(float) - frame["T2M_MIN"].astype(float)).clip(lower=0)
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
    et0_day = (0.0023* (tmean + 17.8)* np.sqrt(trange)* (np.maximum(ra_extraterrestrial, 0) / 2.45))
    return pd.Series(
        np.maximum(et0_day, 0) * frame.index.days_in_month,
        index=frame.index,
        name="pet_mm",
    )


def _tail_slopes(sorted_cal: np.ndarray, z_cal: np.ndarray, k: int = 5) -> tuple[float, float]:
    """Stable dry/wet tail slopes (z per raw unit) for extrapolation."""
    edge_lo = (z_cal[1] - z_cal[0]) / (sorted_cal[1] - sorted_cal[0])
    edge_hi = (z_cal[-1] - z_cal[-2]) / (sorted_cal[-1] - sorted_cal[-2])
    if len(sorted_cal) < k + 1:
        return float(edge_lo), float(edge_hi)
    lo = np.polyfit(sorted_cal[: k + 1], z_cal[: k + 1], 1)[0]
    hi = np.polyfit(sorted_cal[-(k + 1) :], z_cal[-(k + 1) :], 1)[0]
    cap = 3.0 * np.median(np.diff(z_cal) / np.diff(sorted_cal))
    return float(min(max(lo, 0.0), cap)), float(min(max(hi, 0.0), cap))


def _nonparametric_z_scores(values: np.ndarray, calibration: np.ndarray, max_abs_z: float = 3.5) -> np.ndarray:
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
    balance: pd.Series, scale: int, method: str, calibration_end: str = SPEI_CALIBRATION_END
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
    end_month: int = 12,
    daily_monthly: pd.DataFrame | None = None,
    refresh_power: bool = False,
) -> pd.DataFrame:
    frame = fetch_power(site, spec, start, end, cache, end_month, refresh_power)
    frame = frame.loc[frame.index.month != 13]
    expected = pd.date_range(f"{start}-01-01", f"{end}-{end_month:02d}-01", freq="MS")
    frame = frame.reindex(expected)
    frame = frame.replace([-999.0, -99.0, -9999.0, -32768.0], np.nan)
    for column in ("GWETTOP", "GWETROOT"):
        if column not in frame:
            frame[column] = np.nan
    if daily_monthly is not None and site in daily_monthly.index.get_level_values("site"):
        daily_site = daily_monthly.xs(site, level="site").reindex(expected)
        fallback_columns = [*POWER_VARS, "GWETTOP", "GWETROOT"]
        for column in fallback_columns:
            if column in daily_site:
                frame[column] = frame[column].where(frame[column].notna(), daily_site[column])
    for column in ("GWETTOP", "GWETROOT"):
        if column not in frame:
            frame[column] = np.nan
    frame["T2M_RANGE"] = (frame["T2M_MAX"] - frame["T2M_MIN"]).clip(lower=0)
    es = (
        0.6108 * np.exp(17.27 * frame["T2M_MAX"] / (frame["T2M_MAX"] + 237.3))
        + 0.6108 * np.exp(17.27 * frame["T2M_MIN"] / (frame["T2M_MIN"] + 237.3))
    ) / 2
    ea = 0.6108 * np.exp(17.27 * frame["T2MDEW"] / (frame["T2MDEW"] + 237.3))
    frame["VPD"] = (es - ea).clip(lower=0)
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
    max_stale = int(INDEX_ALIGNMENT.get("max_staleness_months", 12))
    for name in ("oni", "dmi", "nao", "pdo"):
        if name not in indices:
            frame[name] = np.nan
            continue
        lag = int(INDEX_ALIGNMENT[name]["lag_months"])
        # Track dates only where this specific index has a real observation.
        # The NOAA series are concatenated on a shared calendar, so using the
        # union index here falsely made missing PDO months look newly observed.
        observed = indices[name].dropna()
        source_values = observed.reindex(frame.index).shift(lag)
        source_dates = pd.Series(observed.index, index=observed.index).reindex(frame.index).shift(lag)
        source_values = source_values.ffill()
        source_dates = source_dates.ffill()
        age = (frame.index.year - pd.DatetimeIndex(source_dates).year) * 12 + (
            frame.index.month - pd.DatetimeIndex(source_dates).month
        )
        stale = pd.isna(source_dates.to_numpy()) | (age > max_stale)
        frame[name] = source_values.where(~stale)
        frame[f"{name}_source_month"] = pd.DatetimeIndex(source_dates).strftime("%Y-%m")
        frame.loc[stale, f"{name}_source_month"] = None
        frame[f"{name}_staleness_months"] = np.where(stale, np.nan, age)
    frame["site"] = site
    frame["region"] = spec["region"]
    return frame.reset_index()


def load_daily_monthly(path: Path, end_year: int, end_month: int) -> pd.DataFrame | None:
    """Aggregate only complete daily months for use as a POWER monthly fallback."""
    if not path.is_file():
        return None
    daily = pd.read_csv(path)
    if not {"date", "site", "PRECTOTCORR", "T2M", "T2MDEW", "WS10M", "T2M_MAX", "T2M_MIN", "ALLSKY_SFC_SW_DWN", "GWETTOP", "GWETROOT"}.issubset(daily.columns):
        raise ValueError(f"Daily NASA POWER source {path} lacks required fields")
    daily["date"] = pd.to_datetime(daily["date"], errors="raise")
    daily["site"] = daily["site"].astype(str).str.strip().str.lower()
    numeric = ["PRECTOTCORR", "T2M", "T2MDEW", "WS10M", "T2M_MAX", "T2M_MIN", "ALLSKY_SFC_SW_DWN", "GWETTOP", "GWETROOT"]
    daily[numeric] = daily[numeric].apply(pd.to_numeric, errors="coerce").replace(
        [-999.0, -99.0, -9999.0, -32768.0], np.nan
    )
    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()
    aggregations = daily.groupby(["site", "month"], sort=False).agg(
        {**{column: "mean" for column in numeric}, "date": "nunique"}
    ).rename(columns={"date": "days_observed"})
    expected_days = aggregations.index.get_level_values("month").days_in_month
    complete = aggregations["days_observed"].to_numpy() == expected_days
    aggregations.loc[~complete, numeric] = np.nan
    last_month = pd.Timestamp(end_year, end_month, 1)
    for site, sub in aggregations.groupby(level="site"):
        if last_month in sub.index.get_level_values("month"):
            row = sub.xs(site, level="site").loc[last_month]
            if int(row["days_observed"]) != last_month.days_in_month:
                raise ValueError(f"Daily POWER data do not cover all of {last_month:%Y-%m} for {site}")
    return aggregations


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


def ingest_speibase_points(computed: pd.DataFrame, points_path: Path) -> pd.DataFrame:
    """Use a previously extracted nearest-cell SPEIbase points artifact."""
    points = pd.read_csv(points_path, parse_dates=["date"])
    reference_cols = [f"spei_{scale}_reference" for scale in (3, 6, 12)]
    missing = {"site", "date", *reference_cols}.difference(points.columns)
    if missing:
        raise ValueError(f"SPEIbase points artifact lacks columns: {sorted(missing)}")
    points["site"] = points["site"].astype(str).str.lower().str.strip()
    reference = points[["site", "date", *reference_cols]].rename(
        columns={f"spei_{scale}_reference": f"spei_{scale}" for scale in (3, 6, 12)}
    )
    if reference.duplicated(["site", "date"]).any():
        raise ValueError("SPEIbase points artifact has duplicate site-month rows")
    left = computed.drop(columns=[f"spei_{scale}" for scale in (3, 6, 12)], errors="ignore")
    result = left.merge(reference, on=["site", "date"], how="left", validate="one_to_one")
    for scale in (3, 6, 12):
        observed = f"spei_{scale}_obs"
        if observed in result:
            result[observed] = result[f"spei_{scale}"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--output",
        help="Output CSV; defaults to data.path from the selected config",
    )
    parser.add_argument("--cache", default="data/raw")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--end-month", type=int)
    parser.add_argument("--refresh-indices", action="store_true")
    parser.add_argument("--refresh-power", action="store_true")
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
    parser.add_argument(
        "--speibase-points", type=Path,
        help="Previously extracted matched monthly nearest-cell SPEIbase artifact",
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
    sites = _normalize_sites(cfg.get("sites"))
    _validate_maharashtra_sites(sites)
    data_cfg = cfg.get("data", {})
    start = args.start if args.start is not None else int(data_cfg.get("start_year", 1981))
    end = args.end if args.end is not None else int(data_cfg.get("end_year", 2024))
    end_month = args.end_month if args.end_month is not None else int(data_cfg.get("end_month", 12))
    cal_years = pd.Timestamp(SPEI_CALIBRATION_END).year - start + 1
    if cal_years < 30:
        warnings.warn(
            f"SPEI calibration baseline is {cal_years} years (<30); use --start 1981 or justify.",
            stacklevel=2,
        )
    if end < start or not 1 <= end_month <= 12:
        raise ValueError("Invalid period: end_year must follow start_year and end_month must be 1..12")
    target_source = str(data_cfg.get("target_source", "local")).lower()
    if target_source not in {"local", "speibase"}:
        raise ValueError("data.target_source must be 'local' or 'speibase'")
    if target_source == "speibase" and args.speibase is None and args.speibase_points is None:
        raise ValueError("data.target_source=speibase requires --speibase DIRECTORY or --speibase-points CSV")
    output = Path(args.output or data_cfg.get("path", "data/monthly_sites.csv"))
    cache = Path(args.cache)
    indices = fetch_indices(f"{start}-01", f"{end}-{end_month:02d}", cache, args.refresh_indices)
    daily_monthly = load_daily_monthly(Path("data/maharashtra_daily_raw.csv"), end, end_month)
    data = pd.concat(
        [
            build_site(
                name,
                spec,
                start,
                end,
                cache,
                args.spei_method,
                indices,
                end_month,
                daily_monthly,
                args.refresh_power,
            )
            for name, spec in sites.items()
        ],
        ignore_index=True,
    )
    configured_features = list(data_cfg.get("features", []))
    missing_features = set(configured_features).difference(data.columns)
    if missing_features:
        raise ValueError(f"Built dataset lacks configured features: {sorted(missing_features)}")
    expected_dates = pd.date_range(f"{start}-01-01", f"{end}-{end_month:02d}-01", freq="MS")
    for site, group in data.groupby("site"):
        actual = pd.DatetimeIndex(pd.to_datetime(group["date"]).sort_values())
        if not actual.equals(expected_dates):
            raise ValueError(f"Monthly coverage is incomplete for {site}")
    if args.speibase:
        from data.validate_speibase import compare_netcdf, speibase_points

        validation = compare_netcdf(data, args.speibase, args.speibase_variable, sites)
        validation_path = output.parent / "speibase_validation.csv"
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
            output.parent / "speibase_validation_points.csv", index=False
        )
        if target_source == "speibase":
            data = ingest_speibase(data, args.speibase, sites, args.speibase_variable)
    elif target_source == "speibase":
        data = ingest_speibase_points(data, args.speibase_points)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    quality = target_quality_report(data)
    validate_target_quality(quality)
    (output.parent / "target_quality.json").write_text(
        json.dumps(quality, indent=2, allow_nan=True), encoding="utf-8"
    )
    power_cache_provenance = {}
    for site in sites:
        cache_sidecar = cache / f"{site}_power_{start}_{end}.metadata.json"
        if cache_sidecar.is_file():
            power_cache_provenance[site] = json.loads(
                cache_sidecar.read_text(encoding="utf-8")
            )
        else:
            power_cache_provenance[site] = {
                "unit_validation": "not verified; cache sidecar absent",
                "retrieved_utc": None,
            }
    (output.parent / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": DATASET_PROTOCOL_VERSION,
                "source": "NASA POWER monthly point with daily-derived fallback for complete months; NOAA CPC/PSL teleconnection indices",
                "source_urls": {
                    "nasa_power_monthly": POWER_URL,
                    **{f"noaa_{name}": url for name, url in NOAA.items()},
                },
                "period": [start, end],
                "latest_complete_month": f"{end}-{end_month:02d}",
                "spei_method": args.spei_method,
                "spei_calibration_period": [start, SPEI_CALIBRATION_END],
                "target_source": target_source,
                "split_boundaries": SPLIT_BOUNDARIES,
                "pet_method": "Monthly Hargreaves-Samani approximation using POWER T2M, T2M_MAX, and T2M_MIN monthly aggregates",
                "sites": sites,
                "climate_variables": POWER_VARS + ["GWETTOP", "GWETROOT"],
                "teleconnection_indices": list(NOAA),
                "model_features": configured_features,
                "index_alignment": INDEX_ALIGNMENT,
                "index_last_source_month": {name: str(indices[name].dropna().index.max().date()) if name in indices and indices[name].notna().any() else None for name in NOAA},
                "index_max_staleness_months": INDEX_ALIGNMENT.get("max_staleness_months", 12),
                "index_cache_provenance": indices.attrs.get("provenance", {}),
                "power_cache_provenance": power_cache_provenance,
                "input_units": {
                    "PRECTOTCORR": "mm/day, converted to monthly total by multiplying calendar days",
                    "temperature": "degC", "WS10M": "m/s",
                    "ALLSKY_SFC_SW_DWN": "daily radiation energy; fresh response unit and legacy-cache verification status are recorded per site",
                },
                "dataset_built_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "speibase": str(args.speibase or args.speibase_points) if (args.speibase or args.speibase_points) else None,
                "target_quality_report": str(output.parent / "target_quality.json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(data)} site-month rows to {output}")


if __name__ == "__main__":
    main()
