"""Provider-response checks for future independent-evaluation data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from data.protocol import INDEX_ALIGNMENT, MODEL_FEATURES

POWER_VARIABLES = (
    "PRECTOTCORR", "T2M", "T2MDEW", "WS10M", "GWETTOP", "GWETROOT",
    "T2M_RANGE", "VPD",
)


@dataclass(frozen=True)
class AvailabilityResult:
    site: str
    latest_month: str | None
    missing_variables: tuple[str, ...]
    invalid_rows: int
    complete: bool


def inspect_power_frame(site: str, frame: pd.DataFrame) -> AvailabilityResult:
    """Assess provider data without filling, extrapolating, or writing data."""
    data = frame.copy()
    if "date" in data:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
    missing = tuple(sorted(set(POWER_VARIABLES).difference(data.columns)))
    invalid = 0
    if not data.empty:
        values = data.reindex(columns=POWER_VARIABLES)
        invalid = int((~np.isfinite(values.apply(pd.to_numeric, errors="coerce"))).any(axis=1).sum())
        latest = data["date"].dropna().max()
    else:
        latest = pd.NaT
    return AvailabilityResult(
        site=site,
        latest_month=None if pd.isna(latest) else latest.strftime("%Y-%m"),
        missing_variables=missing,
        invalid_rows=invalid,
        complete=not missing and invalid == 0 and not data.empty,
    )


def inspect_climate_indices(
    index_months: dict[str, str | None], issue_month: str
) -> list[dict[str, Any]]:
    """Apply protocol lag and age rules; stale or missing values remain unavailable."""
    issue = pd.Period(issue_month, freq="M")
    rows = []
    for name, policy in INDEX_ALIGNMENT.items():
        if not isinstance(policy, dict):
            continue
        source = index_months.get(name)
        age = None if source is None else issue - pd.Period(source, freq="M")
        age_months = None if age is None else age.n
        available = age_months is not None and age_months >= policy["publication_lag_months"]
        usable = available and age_months <= policy["max_age_months"]
        rows.append({
            "index": name,
            "source_month": source,
            "publication_lag_months": policy["publication_lag_months"],
            "max_age_months": policy["max_age_months"],
            "age_months": age_months,
            "usable": bool(usable),
            "status": "usable" if usable else ("missing" if source is None else "stale_or_not_published"),
        })
    return rows


def validate_candidate_frame(frame: pd.DataFrame, sites: set[str]) -> list[str]:
    """Return blocking quality issues for a candidate table."""
    issues = []
    required = {"date", "site", *MODEL_FEATURES, "spei_3", "spei_6", "spei_12"}
    issues.extend(f"missing_column:{column}" for column in sorted(required.difference(frame.columns)))
    if {"site", "date"}.issubset(frame.columns):
        if frame.duplicated(["site", "date"]).any():
            issues.append("duplicate_site_month")
        if set(frame["site"].astype(str)) != sites:
            issues.append("site_coverage_incomplete")
    return issues
