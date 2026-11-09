"""Reusable temporal split and leakage validation for Phase 5.1."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TemporalRequirements:
    seq_len: int
    forecast_steps: int
    accumulation_windows: tuple[int, ...] = (3, 6, 12)
    preprocessing_lag_months: int = 0

    @property
    def max_accumulation(self) -> int:
        return max(self.accumulation_windows)

    @property
    def required_embargo_months(self) -> int:
        return (
            self.seq_len
            + self.forecast_steps
            + self.max_accumulation
            + self.preprocessing_lag_months
        )


def _dates(values: Iterable[Any], label: str) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(list(values), errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{label} contains missing or invalid dates")
    return pd.DatetimeIndex(parsed).to_period("M").to_timestamp()


def validate_samples(
    samples: Iterable[dict[str, Any]],
    split_boundaries: dict[str, str],
    requirements: TemporalRequirements,
) -> dict[str, Any]:
    """Validate window metadata and return auditable counts and violations."""
    rows = list(samples)
    boundaries = {key: pd.Timestamp(value).to_period("M").to_timestamp()
                  for key, value in split_boundaries.items()}
    order = ("train", "val", "test")
    if not (
        boundaries["train_end"] < boundaries["val_start"]
        <= boundaries["val_end"] < boundaries["test_start"]
        <= boundaries["test_end"]
    ):
        raise ValueError("Split boundaries are not strictly chronological")

    violations: list[dict[str, Any]] = []
    origins: list[tuple[str, str, str]] = []
    combinations: list[tuple[str, str, str]] = []
    counts = Counter()
    for index, sample in enumerate(rows):
        site = str(sample.get("site", "")).strip()
        split = str(sample.get("split", ""))
        input_dates = _dates(sample.get("input_dates", []), f"sample {index} input_dates")
        target_dates = _dates(sample.get("target_dates", []), f"sample {index} target_dates")
        if not site or split not in order or not len(input_dates) or not len(target_dates):
            violations.append({"sample": index, "reason": "missing site, split, or dates"})
            continue
        if not input_dates.is_monotonic_increasing or not target_dates.is_monotonic_increasing:
            violations.append({"sample": index, "reason": "dates are not chronological"})
        if set(input_dates).intersection(target_dates):
            violations.append({"sample": index, "reason": "input/target overlap"})
        if target_dates[0] <= input_dates[-1]:
            violations.append({"sample": index, "reason": "target starts before input ends"})
        target_values = sample.get("target_values")
        if target_values is not None and (
            len(target_values) != len(target_dates)
            or any(value is None or pd.isna(value) for value in target_values)
        ):
            violations.append({"sample": index, "reason": "missing site-month target"})
        expected = boundaries[f"{split}_end"] if split != "val" else boundaries["val_end"]
        start = boundaries["train_end"] if split == "train" else boundaries[f"{split}_start"]
        if input_dates[0] < start or target_dates[-1] > expected:
            violations.append({"sample": index, "reason": "sample crosses split boundary"})
        origin = (site, input_dates[-1].strftime("%Y-%m"), split)
        origins.append(origin)
        for date in target_dates:
            combinations.append((site, date.strftime("%Y-%m"), str(sample.get("target", "all"))))
        counts[split] += 1

    duplicate_origins = [key for key, count in Counter(origins).items() if count > 1]
    duplicate_combinations = [key for key, count in Counter(combinations).items() if count > 1]
    if duplicate_origins:
        violations.append({"reason": "duplicate forecast origins", "values": duplicate_origins})
    if duplicate_combinations:
        violations.append({"reason": "duplicate site-month-target combinations",
                           "values": duplicate_combinations})

    gaps = {
        "train_to_val": (boundaries["val_start"].year - boundaries["train_end"].year) * 12
        + boundaries["val_start"].month - boundaries["train_end"].month,
        "val_to_test": (boundaries["test_start"].year - boundaries["val_end"].year) * 12
        + boundaries["test_start"].month - boundaries["val_end"].month,
    }
    embargo = {name: {"actual_months": gap,
                      "required_months": requirements.required_embargo_months,
                      "pass": gap >= requirements.required_embargo_months}
               for name, gap in gaps.items()}
    return {
        "sample_counts": dict(counts),
        "total_samples": len(rows),
        "duplicate_forecast_origins": duplicate_origins,
        "duplicate_site_month_target": duplicate_combinations,
        "violations": violations,
        "embargo": embargo,
        "pass": not violations and all(item["pass"] for item in embargo.values()),
    }
