"""Reconcile canonical October 2026 values with POWER monthly and daily sources."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

from data.api import PARAMETERS as DAILY_PARAMETERS
from data.api import POWER_API_URL as DAILY_POWER_URL
from data.build_dataset import POWER_URL, POWER_VARS

AUDIT_PERIOD = "2026-10"
OUTPUT = Path("results/maharashtra/phase1/power_source_reconciliation.json")


def classify_october_value(
    existing: float | None,
    monthly_value: float | None,
    fresh_daily_value: float | None,
    daily_complete: bool,
    *,
    tolerance: float = 1e-6,
) -> tuple[str, str]:
    """Classify canonical values using direct evidence, never imputation."""
    if existing is None:
        if monthly_value is not None or (daily_complete and fresh_daily_value is not None):
            return "provisional", "A fresh source value exists, but the canonical October value is missing."
        return "unavailable", "No canonical value or usable fresh source value is available."
    if monthly_value is not None:
        if np.isclose(existing, monthly_value, atol=tolerance, rtol=0):
            return "verified", "Canonical value matches the October value in the NASA POWER monthly response."
        return "unverified", "Canonical value differs from the available monthly response."
    if daily_complete and fresh_daily_value is not None:
        if np.isclose(existing, fresh_daily_value, atol=tolerance, rtol=0):
            return "provisional", "Canonical daily-fallback value matches a fresh complete daily query; the monthly product has no October value and the original daily-cache request provenance is absent."
        return "unverified", "Fresh daily query is complete but does not match the canonical daily-fallback value."
    return "unverified", "October value exists, but its original source cannot be established and no matching fresh source value is available."


def _finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _fresh_daily_query(site: str, spec: dict) -> tuple[pd.DataFrame | None, dict]:
    retrieved = datetime.now(UTC).isoformat()
    params = {
        "parameters": ",".join(DAILY_PARAMETERS),
        "community": "ag",
        "longitude": spec["lon"],
        "latitude": spec["lat"],
        "start": 20261001,
        "end": 20261031,
        "format": "json",
        "units": "metric",
    }
    metadata = {
        "url": DAILY_POWER_URL,
        "params": params,
        "retrieved_utc": retrieved,
        "http_status": None,
    }
    try:
        response = requests.get(DAILY_POWER_URL, params=params, timeout=90)
        metadata["http_status"] = response.status_code
        response.raise_for_status()
        body = response.json()
        pdata = body["properties"]["parameter"]
        frame = pd.DataFrame(pdata)
        frame.index = pd.to_datetime(frame.index, format="%Y%m%d", errors="raise")
        frame.index.name = "date"
        frame = frame.apply(pd.to_numeric, errors="coerce").replace(
            [-999.0, -99.0, -9999.0, -32768.0], np.nan
        )
        metadata.update({
            "header": body.get("header", {}),
            "units": {k: v.get("units") if isinstance(v, dict) else v for k, v in body.get("parameters", {}).items()},
            "returned_start": frame.index.min().strftime("%Y-%m-%d") if len(frame) else None,
            "returned_end": frame.index.max().strftime("%Y-%m-%d") if len(frame) else None,
            "returned_days": int(frame.index.nunique()),
        })
        return frame, metadata
    except (requests.RequestException, ValueError, KeyError) as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        return None, metadata


def run_audit() -> dict:
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    monthly = pd.read_csv(cfg["data"]["path"], parse_dates=["date"])
    daily_path = Path("data/maharashtra_daily_raw.csv")
    daily_cache = pd.read_csv(daily_path, parse_dates=["date"])
    daily_cache["site"] = daily_cache["site"].astype(str).str.lower()
    month_start = pd.Timestamp(f"{AUDIT_PERIOD}-01")
    month_end = month_start + pd.offsets.MonthEnd(0)
    records = []
    site_fresh = {}

    for site, spec in cfg["sites"].items():
        site_data = monthly.loc[monthly.site.astype(str).str.lower().eq(site)]
        row = site_data.loc[site_data.date.eq(month_start)]
        if len(row) != 1:
            raise ValueError(f"Expected one canonical October row for {site}; got {len(row)}")
        canonical = row.iloc[0]
        cached_oct = daily_cache.loc[
            daily_cache.site.eq(site)
            & daily_cache.date.between(month_start, month_end)
        ].sort_values("date")
        expected_days = pd.date_range(month_start, month_end, freq="D")
        daily_cache_dates = pd.DatetimeIndex(cached_oct.date.drop_duplicates().sort_values())
        daily_cache_complete = daily_cache_dates.equals(expected_days)
        monthly_cache_path = Path(f"data/raw_protocol7_refresh/{site}_power_1990_2026_v10.csv")
        monthly_meta_path = monthly_cache_path.with_suffix(".metadata.json")
        monthly_cache = pd.read_csv(monthly_cache_path, index_col=0, parse_dates=True)
        monthly_meta = json.loads(monthly_meta_path.read_text(encoding="utf-8"))
        monthly_oct = monthly_cache.loc[month_start] if month_start in monthly_cache.index else pd.Series(dtype=float)
        fresh, fresh_meta = _fresh_daily_query(site, spec)
        site_fresh[site] = fresh_meta
        fresh_oct = fresh.loc[fresh.index.to_period("M") == month_start.to_period("M")] if fresh is not None else None
        fresh_complete = bool(
            fresh_oct is not None
            and pd.DatetimeIndex(fresh_oct.index.unique()).sort_values().equals(expected_days)
        )

        for variable in POWER_VARS:
            existing = _finite_float(canonical.get(variable))
            monthly_value = _finite_float(monthly_oct.get(variable))
            daily_cached_value = None
            if daily_cache_complete and variable in cached_oct:
                values = pd.to_numeric(cached_oct[variable], errors="coerce").replace(
                    [-999.0, -99.0, -9999.0, -32768.0], np.nan
                )
                if values.notna().all():
                    daily_cached_value = float(values.mean())
            fresh_value = None
            fresh_var_complete = False
            if fresh_complete and fresh_oct is not None and variable in fresh_oct:
                fresh_values = pd.to_numeric(fresh_oct[variable], errors="coerce").replace(
                    [-999.0, -99.0, -9999.0, -32768.0], np.nan
                )
                fresh_var_complete = bool(fresh_values.notna().all())
                if fresh_var_complete:
                    fresh_value = float(fresh_values.mean())
            classification, reason = classify_october_value(
                existing,
                monthly_value,
                fresh_value,
                fresh_complete and fresh_var_complete,
            )
            records.append({
                "site": site,
                "variable": variable,
                "month": AUDIT_PERIOD,
                "existing_october_value": existing,
                "cached_daily_aggregate": daily_cached_value,
                "monthly_cache_october_value": monthly_value,
                "fresh_daily_monthly_mean": fresh_value,
                "cached_daily_complete": daily_cache_complete,
                "fresh_daily_complete_for_variable": fresh_complete and fresh_var_complete,
                "monthly_cache": {
                    "path": str(monthly_cache_path),
                    "source_url": POWER_URL,
                    "retrieved_utc": monthly_meta.get("retrieved_utc"),
                    "header_start": monthly_meta.get("header", {}).get("start"),
                    "header_end": monthly_meta.get("header", {}).get("end"),
                    "unit_validation": monthly_meta.get("unit_validation"),
                    "units": monthly_meta.get("units", {}).get(variable),
                    "october_value_present": monthly_value is not None,
                },
                "daily_cache": {
                    "path": str(daily_path),
                    "file_mtime_utc": datetime.fromtimestamp(daily_path.stat().st_mtime, UTC).isoformat(),
                    "original_source_url": None,
                    "original_request_params": None,
                    "original_retrieval_timestamp": None,
                    "note": "Builder code identifies this file as daily NASA POWER fallback input; the file itself contains no request sidecar/provenance.",
                },
                "fresh_query": fresh_meta,
                "classification": classification,
                "reason": reason,
                "evidence": [
                    "Canonical monthly table row",
                    "Builder replaces monthly missing values with means from complete daily calendar months",
                    "Cached monthly POWER October cell and retrieval sidecar",
                    "Cached daily file October date/value completeness",
                    "Fresh daily POWER query and matched monthly aggregation when available",
                ],
            })

    return {
        "generated_utc": datetime.now(UTC).isoformat(),
        "period": AUDIT_PERIOD,
        "canonical_dataset": cfg["data"]["path"],
        "site_count": len(cfg["sites"]),
        "variables": POWER_VARS,
        "fresh_query_by_site": site_fresh,
        "records": records,
        "limitations": [
            "The daily cache has no original NASA request URL, parameters, or retrieval timestamp; filesystem modification time is recorded separately and is not treated as a retrieval timestamp.",
            "Monthly POWER cache metadata records requested coverage through 2026-11 but its October data cells are missing; the header end date is not proof of returned valid monthly observations.",
        ],
    }


def main() -> None:
    report = run_audit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    counts = pd.Series([r["classification"] for r in report["records"]]).value_counts().to_dict()
    print(f"Wrote {OUTPUT}; classification counts: {counts}")


if __name__ == "__main__":
    main()
