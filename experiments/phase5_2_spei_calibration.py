"""Origin-aware SPEI calibration helpers for Phase 5.2 experiments."""

from __future__ import annotations

import pandas as pd

from data.build_dataset import _standardize_accumulation


def origin_aware_standardize(
    balance: pd.Series,
    scale: int,
    method: str,
    origin: pd.Timestamp,
    calibration_start: pd.Timestamp,
) -> pd.Series:
    """Transform scores using only calibration observations through ``origin``."""
    origin = pd.Timestamp(origin).to_period("M").to_timestamp()
    start = pd.Timestamp(calibration_start).to_period("M").to_timestamp()
    if start > origin:
        raise ValueError("calibration_start cannot be after origin")
    return _standardize_accumulation(
        balance,
        scale,
        method,
        calibration_end=str(origin.date()),
        calibration_start=str(start.date()),
    )
