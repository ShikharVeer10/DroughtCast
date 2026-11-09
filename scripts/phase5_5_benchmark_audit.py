"""Create the Phase 5.5 pre-experiment audit without fitting models."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.protocol import PHASE5_1_1_SPLITS, SPEI_CALIBRATION_END

OUT = ROOT / "results" / "maharashtra" / "phase5" / "phase5_5_benchmarking"
DATA = ROOT / "data" / "maharashtra" / "monthly_sites.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_windows(frame: pd.DataFrame, start: str | None, end: str) -> int:
    count = 0
    for _, group in frame.groupby("site"):
        dates = group.sort_values("date")["date"].tolist()
        for index in range(len(dates) - 9):
            window = dates[index:index + 10]
            if (start is None or window[0] >= pd.Timestamp(start)) and window[-1] <= pd.Timestamp(end):
                count += 1
    return count


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA, parse_dates=["date"])
    partitions = {
        "training": [str(frame.date.min().date()), PHASE5_1_1_SPLITS["train_end"]],
        "validation": [PHASE5_1_1_SPLITS["val_start"], PHASE5_1_1_SPLITS["val_end"]],
        "final_evaluation": [PHASE5_1_1_SPLITS["test_start"], PHASE5_1_1_SPLITS["test_end"]],
    }
    counts = {
        role: count_windows(frame, bounds[0], bounds[1])
        for role, bounds in partitions.items()
    }
    (OUT / "experiment_configuration.json").write_text(json.dumps({
        "status": "BLOCKED_BEFORE_MODEL_FITTING",
        "models_planned": ["persistence", "seasonal_climatology", "ridge", "xgboost_or_existing_tree", "shared_lstm"],
        "splits": PHASE5_1_1_SPLITS,
        "spei_fixed_calibration_end": SPEI_CALIBRATION_END,
        "embargo_months": 22,
        "probabilistic_calibration": "disabled; Phase 5.3 calibration partition has zero windows",
        "test_period_used_for_selection": True,
        "model_fitting_performed": False,
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "pre_experiment_contamination_audit.md").write_text(
        "# Pre-experiment contamination audit\n\n"
        "The proposed 2017–2025 final period is **not untouched**. `LEAKAGE_AUDIT.md` explicitly "
        "references existing seed-42 and multi-seed test metrics, and historical Phase 2/3 reports "
        "inspect test results and development decisions. It must not be described as an independent "
        "confirmatory test. No new model fitting was performed because this phase scope forbids training "
        "or tuning; a future descriptive benchmark must label the period contaminated.\n\n"
        f"Fixed SPEI targets are valid only for origins on or after {SPEI_CALIBRATION_END}. "
        "The Phase 5.4 manifest contains 39 eligible development origins and 504 unique site-month "
        "target observations; overlapping windows are dependent.\n", encoding="utf-8")
    pd.DataFrame([
        {"partition": role, "start": bounds[0], "end": bounds[1], "complete_window_count": counts[role],
         "role": {"training": "parameter fitting", "validation": "model selection/early stopping",
                  "final_evaluation": "descriptive final-period evaluation only"}[role]}
        for role, bounds in partitions.items()
    ]).to_csv(OUT / "sample_accounting.csv", index=False)
    (OUT / "model_inventory.json").write_text(json.dumps({
        "planned": [
            {"name": "persistence", "learned": False},
            {"name": "seasonal_climatology", "learned": True, "fit_scope": "training only"},
            {"name": "ridge", "learned": True, "fit_scope": "training only"},
            {"name": "existing_tree", "learned": True, "fit_scope": "training only"},
            {"name": "shared_lstm", "learned": True, "fit_scope": "training/validation only"},
        ],
        "executed": [],
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "limitations_report.md").write_text(
        "# Limitations\n\n"
        "- No prediction-level outputs or metrics exist because model fitting is prohibited in this phase.\n"
        "- The 2017–2025 period is previously inspected and cannot support untouched-test claims.\n"
        "- No independent calibration partition exists; calibrated coverage is not reported.\n"
        "- Rolling-origin windows overlap in site and target month; ordinary row-wise significance testing is invalid.\n",
        encoding="utf-8")
    (OUT / "verification_report.md").write_text(
        "# Verification report\n\n"
        "**BLOCKED.** The audit completed without model execution. The benchmark must be redesigned "
        "with an uninspected evaluation period and explicit authorization to fit models before it can "
        "produce scientifically interpretable comparisons.\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.name != "sha256_manifest.json")
    (OUT / "sha256_manifest.json").write_text(
        json.dumps({path.name: sha256(path) for path in files}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
