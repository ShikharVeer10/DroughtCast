"""Deterministic, read-only rolling-origin design for Phase 5.4."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from data.protocol import PHASE5_1_1_SPLITS, SPEI_CALIBRATION_END


def generate_rolling_origins(
    frame: pd.DataFrame,
    *,
    origins: Iterable[pd.Timestamp] | None = None,
    seq_len: int = 6,
    forecast_steps: int = 4,
    embargo_months: int = 22,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return deterministic sample rows and explicit exclusions."""
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().any():
        raise ValueError("Invalid dates in rolling-origin input")
    if data.duplicated(["site", "date"]).any():
        raise ValueError("Duplicate site-month observations")
    if origins is None:
        origins = pd.date_range(
            PHASE5_1_1_SPLITS["val_start"], PHASE5_1_1_SPLITS["val_end"], freq="MS"
        )
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    features = list(features or [])
    for origin_value in sorted(pd.Timestamp(value) for value in origins):
        origin = origin_value.to_period("M").to_timestamp()
        train_end = origin - pd.DateOffset(months=embargo_months)
        validation_start = train_end + pd.DateOffset(months=embargo_months)
        validation_end = origin
        if origin < pd.Timestamp(SPEI_CALIBRATION_END):
            exclusions.append({"origin": origin.strftime("%Y-%m"), "reason": "invalid SPEI calibration origin"})
            continue
        target_end = origin + pd.DateOffset(months=forecast_steps)
        if target_end >= pd.Timestamp(PHASE5_1_1_SPLITS["test_start"]):
            exclusions.append({"origin": origin.strftime("%Y-%m"), "reason": "overlaps final evaluation period"})
            continue
        for site, group in data.groupby("site", sort=True):
            group = group.sort_values("date").set_index("date")
            input_dates = pd.date_range(origin - pd.DateOffset(months=seq_len - 1), origin, freq="MS")
            target_dates = pd.date_range(origin + pd.DateOffset(months=1), target_end, freq="MS")
            missing = input_dates.union(target_dates).difference(group.index)
            if len(missing):
                exclusions.append({"origin": origin.strftime("%Y-%m"), "site": site,
                                   "reason": "missing monthly observation", "missing_count": len(missing)})
                continue
            inputs = group.loc[input_dates]
            targets = group.loc[target_dates]
            if features and not inputs[features].apply(pd.to_numeric, errors="coerce").notna().all().all():
                exclusions.append({"origin": origin.strftime("%Y-%m"), "site": site,
                                   "reason": "missing input feature"})
                continue
            for scale in ("spei_3", "spei_6", "spei_12"):
                if not pd.to_numeric(targets[scale], errors="coerce").notna().all():
                    exclusions.append({"origin": origin.strftime("%Y-%m"), "site": site,
                                       "scale": scale, "reason": "missing SPEI target"})
                    continue
                for lead, target_date in enumerate(target_dates, start=1):
                    rows.append({
                        "origin": origin.strftime("%Y-%m"),
                        "training_start": data["date"].min().strftime("%Y-%m"),
                        "training_end": train_end.strftime("%Y-%m"),
                        "validation_start": validation_start.strftime("%Y-%m"),
                        "validation_end": validation_end.strftime("%Y-%m"),
                        "embargo_months": embargo_months,
                        "site": str(site),
                        "input_start": input_dates[0].strftime("%Y-%m"),
                        "input_end": input_dates[-1].strftime("%Y-%m"),
                        "target_date": target_date.strftime("%Y-%m"),
                        "scale": scale,
                        "lead": lead,
                        "valid_sample": True,
                    })
    return pd.DataFrame(rows), pd.DataFrame(exclusions)
