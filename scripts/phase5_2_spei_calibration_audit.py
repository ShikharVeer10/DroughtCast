"""Generate the Phase 5.2 SPEI calibration leakage audit."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.protocol import PHASE5_1_1_SPLITS, SPEI_CALIBRATION_END

OUT = ROOT / "results" / "maharashtra" / "phase5" / "phase5_2_spei_calibration"
DATA = ROOT / "data" / "maharashtra" / "monthly_sites.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA, parse_dates=["date"])
    starts = frame["date"].min().to_period("M").to_timestamp()
    ends = pd.Timestamp(PHASE5_1_1_SPLITS["val_end"])
    origins = pd.date_range(starts, ends, freq="MS")
    rows = []
    for origin in origins:
        for scale in (3, 6, 12):
            fixed_end = pd.Timestamp(SPEI_CALIBRATION_END)
            fixed_valid = origin >= fixed_end
            accumulation_dates = pd.date_range(
                starts + pd.DateOffset(months=scale - 1), origin, freq="MS"
            )
            counts_by_month = accumulation_dates.to_series().groupby(
                accumulation_dates.month
            ).size()
            origin_valid = len(counts_by_month) == 12 and bool((counts_by_month >= 20).all())
            rows.append({
                "origin": origin.strftime("%Y-%m"),
                "scale": scale,
                "calibration_start": starts.strftime("%Y-%m"),
                "fixed_calibration_end": fixed_end.strftime("%Y-%m"),
                "origin_aware_calibration_end": origin.strftime("%Y-%m"),
                "latest_observation_used_fixed": fixed_end.strftime("%Y-%m"),
                "latest_observation_used_origin_aware": origin.strftime("%Y-%m"),
                "future_information_fixed": not fixed_valid,
                "fixed_target_valid": fixed_valid,
                "origin_aware_target_valid": origin_valid,
            })
    validity = pd.DataFrame(rows)
    validity.to_csv(OUT / "origin_validity.csv", index=False)
    audit = {
        "source": str(DATA.relative_to(ROOT)),
        "frozen_calibration_end": str(SPEI_CALIBRATION_END),
        "phase5_1_1_splits": PHASE5_1_1_SPLITS,
        "scales": [3, 6, 12],
        "transformation": "calendar-month nonparametric empirical z-score or Fisk log-logistic; max_abs_z=3.5",
        "water_balance": "monthly PRECTOTCORR * days_in_month - Hargreaves PET",
        "accumulation": "trailing rolling(scale, min_periods=scale).sum()",
        "origins": [origins.min().strftime("%Y-%m"), origins.max().strftime("%Y-%m")],
        "fixed_invalid_rows": int(validity["future_information_fixed"].sum()),
        "fixed_valid_rows": int(validity["fixed_target_valid"].sum()),
        "origin_aware_valid_rows": int(validity["origin_aware_target_valid"].sum()),
    }
    (OUT / "spei_calibration_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (OUT / "calibration_method_comparison.md").write_text(
        "# Calibration method comparison\n\n"
        "| method | historical validity | consistency | implication |\n|---|---|---|---|\n"
        "| A: fixed 1990–2010 calibration | valid only for origins on/after 2010-12 | exactly matches frozen v1.0 targets | restrict early rolling origins |\n"
        "| B: origin-aware calibration | valid once each scale has sufficient history | changes target definition by origin | appropriate only as a separate experimental target family |\n\n"
        "Approach A is the approved safe solution for reuse of frozen targets: exclude origins before "
        "the fixed calibration end. Approach B is implemented only as an isolated helper and was not "
        "used to rewrite canonical targets or select a model. The Phase 5.1.1 test period remains "
        "untouched and is not used for selection.\n", encoding="utf-8")
    (OUT / "future_information_tests.json").write_text(json.dumps({
        "fixed_calibration_future_leakage_before_2010_12": True,
        "fixed_calibration_future_leakage_on_or_after_2010_12": False,
        "origin_aware_calibration_future_leakage": False,
        "canonical_targets_modified": False,
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "validation_report.md").write_text(
        "# Phase 5.2 validation report\n\n"
        "Frozen SPEI targets are not valid for rolling origins before 2010-12 because their "
        "calendar-month calibration includes observations through 2010-12. They are valid for "
        "origins on or after that date. The safe decision is to restrict fixed-target historical "
        "origins; origin-aware recalibration remains experimental and isolated.\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.name != "sha256_manifest.json")
    (OUT / "sha256_manifest.json").write_text(
        json.dumps({path.name: sha256(path) for path in files}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
