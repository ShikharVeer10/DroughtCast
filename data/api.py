"""Local FastAPI service for fetching/importing NASA POWER daily data.

Run with: uvicorn data.api:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from pydantic import BaseModel, Field
except ImportError as exc:  # Give a useful error when optional API packages are absent.
    raise RuntimeError("FastAPI service requires `pip install fastapi uvicorn[standard]`.") from exc

from data.build_dataset import NOAA, SITES, _parse_noaa_text

app = FastAPI(title="DroughtCast NASA POWER Data API", version="1.0.0")
DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = ["PRECTOTCORR", "T2M", "T2MDEW", "WS10M", "ALLSKY_SFC_SW_DWN", "T2M_MAX", "T2M_MIN"]
RAW_DIR = Path("data/raw/daily")
OUTPUT = Path("data/monthly_sites.csv")


class BuildRequest(BaseModel):
    start: int = Field(19950101, ge=19810101)
    end: int = Field(20241231, ge=19810101)
    sites: list[str] = list(SITES)
    spei_method: str = "nonparametric"


def fetch_daily(site: str, spec: dict[str, Any], start: int, end: int) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{site}_{start}_{end}.csv"
    uploaded_path = RAW_DIR / f"{site}_uploaded.csv"
    if uploaded_path.exists() and not path.exists():
        uploaded = pd.read_csv(uploaded_path, parse_dates=["date"]).set_index("date")
        lo = pd.to_datetime(str(start), format="%Y%m%d")
        hi = pd.to_datetime(str(end), format="%Y%m%d")
        return uploaded.loc[lo:hi]
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"]).set_index("date")
    start_dt = pd.to_datetime(str(start), format="%Y%m%d")
    end_dt = pd.to_datetime(str(end), format="%Y%m%d")
    chunks = []
    for year in range(start_dt.year, end_dt.year + 1):
        chunk_start = max(start_dt, pd.Timestamp(year, 1, 1))
        chunk_end = min(end_dt, pd.Timestamp(year, 12, 31))
        cstart, cend = int(chunk_start.strftime("%Y%m%d")), int(chunk_end.strftime("%Y%m%d"))
        try:
            response = requests.get(DAILY_URL, params={
                "start": cstart, "end": cend, "latitude": spec["lat"], "longitude": spec["lon"],
                "community": "AG", "parameters": ",".join(PARAMETERS), "format": "JSON",
                "units": "metric", "time-standard": "LST",
            }, timeout=180)
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"NASA POWER request failed for {site} ({year}): {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=f"NASA POWER returned invalid JSON for {site} ({year})") from exc
        try:
            parameters = body["properties"]["parameter"]
            missing = set(PARAMETERS).difference(parameters)
            if missing:
                raise ValueError(f"NASA response omitted parameters: {sorted(missing)}")
            raw_chunk = pd.DataFrame(parameters)
            raw_chunk.index = pd.to_datetime(raw_chunk.index, format="%Y%m%d")
            raw_chunk = raw_chunk.apply(pd.to_numeric, errors="coerce").replace(-999, np.nan)
            chunks.append(raw_chunk)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=f"Unexpected NASA POWER response for {site} ({year}): {exc}") from exc
    raw = pd.concat(chunks).sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    raw.index.name = "date"
    raw.to_csv(path, index_label="date")
    return raw


def pet_daily_aggregate(daily: pd.DataFrame, latitude: float) -> pd.DataFrame:
    """Aggregate daily POWER data; calculate Hargreaves PET from daily extrema.

    ET0 (mm/day) = 0.0023 * (Tmean + 17.8) * sqrt(TR) * (Ra / 2.45), matching
    data.build_dataset.estimate_pet_hargreaves.
    """
    idx = daily.index
    tmean = daily["T2M"].astype(float)
    trange = (daily["T2M_MAX"].astype(float) - daily["T2M_MIN"].astype(float)).clip(lower=0)
    phi = np.deg2rad(latitude)
    doy = idx.dayofyear.to_numpy()
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    ra_extraterrestrial = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    et0 = 0.0023 * (tmean + 17.8) * np.sqrt(trange) * (np.maximum(ra_extraterrestrial, 0) / 2.45)
    work = daily.copy()
    # Daily POWER PRECTOTCORR is mm/day; solar irradiance is an energy/day value.
    work["pet_mm"] = et0
    work["PRECTOTCORR"] = work["PRECTOTCORR"].astype(float)
    monthly = work.resample("MS").agg({
        "PRECTOTCORR": "sum", "T2M": "mean", "T2MDEW": "mean", "WS10M": "mean",
        "ALLSKY_SFC_SW_DWN": "mean", "T2M_MAX": "mean", "T2M_MIN": "mean", "pet_mm": "sum",
    })
    counts = work["T2M"].resample("MS").count()
    expected = pd.Series(
        [pd.Period(month, freq="M").days_in_month for month in counts.index.to_period("M")],
        index=counts.index,
    )
    monthly.loc[counts < expected, :] = np.nan
    return monthly


def get_indices(start: str, end: str) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    series = []
    for name, url in NOAA.items():
        path = RAW_DIR.parent / f"noaa_{name}.txt"
        if path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                text = resp.text
            except requests.RequestException as exc:
                raise HTTPException(status_code=502, detail=f"NOAA {name} request failed: {exc}") from exc
            path.write_text(text, encoding="utf-8")
        try:
            series.append(_parse_noaa_text(name, text))
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    return pd.concat(series, axis=1).sort_index().loc[start:end]


def standardize_spei_train_only(balance: pd.Series, scale: int, method: str) -> pd.Series:
    """Fit calendar-month empirical CDFs on data through 2010 only."""
    from data.build_dataset import _nonparametric_z_scores
    from statistics import NormalDist

    accumulation = balance.rolling(scale, min_periods=scale).sum()
    result = pd.Series(np.nan, index=balance.index, dtype=float)
    calibration_end = pd.Timestamp("2010-12-01")
    for month in range(1, 13):
        calibration = accumulation.loc[(accumulation.index.month == month) &
                                       (accumulation.index <= calibration_end)].dropna()
        values = accumulation.loc[accumulation.index.month == month]
        if len(calibration) < 20:
            continue
        if method == "nonparametric":
            z = _nonparametric_z_scores(values.to_numpy(dtype=float), calibration.to_numpy(dtype=float))
            result.loc[values.index] = z
        else:
            sample = calibration.to_numpy(dtype=float)
            location = sample.min() - max(sample.std() * 1e-3, 1e-6)
            logs = np.log(sample - location)
            shape = max(1.0, np.pi / (np.sqrt(3) * max(logs.std(ddof=1), 1e-6)))
            scale_parameter = np.exp(logs.mean())
            shifted = np.maximum(values.to_numpy() - location, 0.0)
            probs = 1.0 / (1.0 + np.maximum(shifted / scale_parameter, 1e-12) ** (-shape))
            result.loc[values.index] = pd.Series(np.clip(probs, 1e-6, 1 - 1e-6), index=values.index).map(NormalDist().inv_cdf)
    return result


def prepare_site(site: str, spec: dict[str, Any], daily: pd.DataFrame,
                 indices: pd.DataFrame, method: str) -> pd.DataFrame:
    monthly = pet_daily_aggregate(daily, spec["lat"])
    # The API aggregation gives precipitation in mm/month; PET is also mm/month.
    balance = monthly["PRECTOTCORR"] - monthly["pet_mm"]
    for scale in (3, 6, 12):
        monthly[f"spei_{scale}"] = standardize_spei_train_only(balance, scale, method)
        monthly[f"spei_{scale}_obs"] = monthly[f"spei_{scale}"]
    for variable in PARAMETERS:
        if variable in monthly:
            monthly[f"lag_{variable}"] = monthly[variable].shift(1)
    monthly = monthly.join(indices, how="left")
    # Apply one conservative monthly rule to every teleconnection: the value
    # attached to month m is the published value from m-1. For centered ONI,
    # this prevents the three-month mean from reaching month m or later.
    for name in ("oni", "dmi", "nao", "pdo"):
        if name in monthly.columns:
            monthly[name] = monthly[name].shift(1)
    monthly["site"], monthly["region"] = site, spec["region"]
    monthly.index.name = "date"
    return monthly.reset_index()


@app.get("/")
def health():
    return {"service": app.title, "docs": "/docs", "parameter_ids": PARAMETERS, "configured_sites": SITES}


@app.get("/power/daily/{site}")
def preview_power(site: str, start: int = 19950101, end: int = 19951231):
    if site not in SITES:
        raise HTTPException(404, detail=f"Unknown site {site}; choose one of {list(SITES)}")
    data = fetch_daily(site, SITES[site], start, end)
    return {"site": site, "rows": len(data), "columns": list(data.columns),
            "start": str(data.index.min().date()), "end": str(data.index.max().date()),
            "cached_csv": str(RAW_DIR / f"{site}_{start}_{end}.csv")}


@app.post("/build")
def build(request: BuildRequest):
    if request.end < request.start:
        raise HTTPException(422, detail="end must be on or after start")
    if request.spei_method not in ("nonparametric", "log-logistic"):
        raise HTTPException(422, detail="spei_method must be nonparametric or log-logistic")
    unknown = set(request.sites).difference(SITES)
    if unknown:
        raise HTTPException(422, detail=f"Unknown sites: {sorted(unknown)}")
    start_date = pd.to_datetime(str(request.start), format="%Y%m%d").to_period("M").start_time
    end_date = pd.to_datetime(str(request.end), format="%Y%m%d").to_period("M").start_time
    indices = get_indices(str(start_date.date()), str(end_date.date()))
    tables = []
    for site in request.sites:
        warm_start = max(19810101, int((start_date - pd.DateOffset(months=13)).strftime("%Y%m%d")))
        daily = fetch_daily(site, SITES[site], warm_start, request.end)
        tables.append(prepare_site(site, SITES[site], daily, indices, request.spei_method))
    output = pd.concat(tables, ignore_index=True)
    output = output[(pd.to_datetime(output["date"]) >= start_date) & (pd.to_datetime(output["date"]) <= end_date)]
    required_features = {"PRECTOTCORR", "T2M", "T2MDEW", "WS10M", "ALLSKY_SFC_SW_DWN", "oni", "dmi", "nao", "pdo", "spei_3_obs", "spei_6_obs", "spei_12_obs"}
    missing_features = required_features.difference(output.columns)
    if missing_features:
        raise HTTPException(status_code=500, detail=f"Prepared data missing training features: {sorted(missing_features)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)
    return {"status": "ok", "output": str(OUTPUT), "rows": len(output),
            "sites": request.sites, "parameters": PARAMETERS,
            "date_range": [str(start_date.date()), str(end_date.date())],
            "note": "SPEI is computed from NASA POWER inputs and Hargreaves-Samani PET; not official SPEIbase."}


@app.post("/upload/{site}")
async def upload_power_csv(site: str, file: UploadFile = File(...), spei_method: str = "nonparametric"):
    """Import a NASA POWER daily CSV export for one configured site."""
    if site not in SITES:
        raise HTTPException(404, detail=f"Unknown site {site}; choose one of {list(SITES)}")
    content = await file.read()
    try:
        table = pd.read_csv(StringIO(content.decode("utf-8-sig")), skip_blank_lines=True)
        # POWER CSV may include metadata before the actual column header; detect the date column.
        if not any(col.upper() in ("DATE", "YEAR", "YYYYMMDD") for col in table.columns):
            table = pd.read_csv(StringIO(content.decode("utf-8-sig")), skiprows=header_row(content))
        daily = normalize_uploaded(table)
    except Exception as exc:
        raise HTTPException(422, detail=f"Could not parse NASA POWER CSV: {exc}") from exc
    path = RAW_DIR / f"{site}_uploaded.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(path, index_label="date")
    return {"status": "uploaded", "site": site, "rows": len(daily), "columns": list(daily.columns),
            "cached_csv": str(path), "next": "POST /build to compute the combined monthly training dataset"}


def header_row(content: bytes) -> int:
    lines = content.decode("utf-8-sig").splitlines()
    for i, line in enumerate(lines):
        normalized = line.replace(" ", "").upper()
        if "YYYYMMDD" in normalized or ("YEAR" in normalized and "DOY" in normalized):
            return i
    raise ValueError("Could not find a data header containing YYYYMMDD or YEAR/DOY")


def normalize_uploaded(table: pd.DataFrame) -> pd.DataFrame:
    lookup = {str(c).strip().upper(): c for c in table.columns}
    data = table.rename(columns={c: str(c).strip().upper() for c in table.columns})
    if "YYYYMMDD" in data:
        dates = pd.to_datetime(data.pop("YYYYMMDD").astype(str), format="%Y%m%d")
    elif "DATE" in data:
        dates = pd.to_datetime(data.pop("DATE"))
    elif "YEAR" in data and "DOY" in data:
        dates = pd.to_datetime(data.pop("YEAR").astype(int).astype(str), format="%Y") + pd.to_timedelta(data.pop("DOY").astype(int) - 1, unit="D")
    else:
        raise ValueError("CSV needs YYYYMMDD/DATE or YEAR and DOY date columns")
    missing = set(PARAMETERS).difference(data.columns)
    if missing:
        raise ValueError(f"CSV lacks required parameter columns: {sorted(missing)}")
    data = data[PARAMETERS].apply(pd.to_numeric, errors="coerce").replace(-999, np.nan)
    data.index = dates
    data.index.name = "date"
    return data.sort_index()
