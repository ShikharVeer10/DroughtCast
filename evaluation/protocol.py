"""Leakage-safe role assignment for Phase 5.3 evaluation design."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise
from typing import Any

import pandas as pd

ROLES = ("training", "validation", "calibration", "final_evaluation")


def validate_role_assignments(
    samples: Iterable[dict[str, Any]],
    embargo_months: int = 22,
) -> dict[str, Any]:
    """Reject role overlap, chronology violations, and target-invalid samples."""
    rows = list(samples)
    violations: list[str] = []
    seen: dict[tuple[str, str], str] = {}
    role_ranges: dict[str, list[pd.Timestamp]] = {role: [] for role in ROLES}
    for row in rows:
        role = row.get("role")
        if role not in ROLES:
            violations.append(f"invalid role: {role}")
            continue
        if row.get("spei_target_valid") is False:
            violations.append("invalid SPEI target assigned to primary protocol")
        input_dates = pd.to_datetime(row.get("input_dates", []), errors="coerce")
        target_dates = pd.to_datetime(row.get("target_dates", []), errors="coerce")
        if input_dates.isna().any() or target_dates.isna().any():
            violations.append("invalid sample date")
            continue
        if set(input_dates).intersection(target_dates):
            violations.append("input/target temporal overlap")
        key = (str(row.get("site")), str(input_dates[-1]))
        if key in seen and seen[key] != role:
            violations.append(f"sample assigned to multiple roles: {key}")
        seen[key] = role
        role_ranges[role].extend([input_dates[0], target_dates[-1]])

    ordered = [role for role in ROLES if role_ranges[role]]
    for left, right in pairwise(ordered):
        left_end = max(role_ranges[left])
        right_start = min(role_ranges[right])
        months = (right_start.year - left_end.year) * 12 + right_start.month - left_end.month
        if months < embargo_months:
            violations.append(f"{left}->{right} embargo is {months} months, required {embargo_months}")
    return {
        "roles": {role: len([row for row in rows if row.get("role") == role]) for role in ROLES},
        "violations": violations,
        "pass": not violations,
    }
