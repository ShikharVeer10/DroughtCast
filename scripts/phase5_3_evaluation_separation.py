"""Generate Phase 5.3 role-separation audit outputs without model execution."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.protocol import PHASE5_1_1_SPLITS, SPEI_CALIBRATION_END

OUT = ROOT / "results" / "maharashtra" / "phase5" / "phase5_3_evaluation_separation"
DATA = ROOT / "data" / "maharashtra" / "monthly_sites.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_windows(frame: pd.DataFrame, start: str, end: str) -> int:
    count = 0
    for _, group in frame.groupby("site"):
        dates = group.sort_values("date")["date"].tolist()
        for index in range(len(dates) - 9):
            window = dates[index:index + 10]
            if window[0] >= pd.Timestamp(start) and window[-1] <= pd.Timestamp(end):
                count += 1
    return count


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA, parse_dates=["date"])
    # The only feasible chronology with the repaired contract leaves no complete
    # calibration windows between validation selection and final evaluation.
    ranges = {
        "training": (None, PHASE5_1_1_SPLITS["train_end"]),
        "validation": (PHASE5_1_1_SPLITS["val_start"], "2013-12-01"),
        "calibration": ("2015-10-01", PHASE5_1_1_SPLITS["val_end"]),
        "final_evaluation": (PHASE5_1_1_SPLITS["test_start"], PHASE5_1_1_SPLITS["test_end"]),
    }
    counts = {
        role: count_windows(frame, start or str(frame.date.min().date()), end)
        for role, (start, end) in ranges.items()
    }
    pd.DataFrame([
        {"role": role, "start": start or str(frame.date.min().date()),
         "end": end, "sample_count": counts[role],
         "purpose": {"training": "parameter fitting", "validation": "model selection and early stopping",
                     "calibration": "interval/probability calibration only",
                     "final_evaluation": "untouched final evaluation"}[role]}
        for role, (start, end) in ranges.items()
    ]).to_csv(OUT / "partition_manifest.csv", index=False)
    (OUT / "evaluation_protocol.json").write_text(json.dumps({
        "embargo_months": 22,
        "spei_fixed_calibration_end": SPEI_CALIBRATION_END,
        "primary_target_rule": "only targets valid for origins on or after fixed SPEI calibration end",
        "roles": ranges,
        "calibration_method": "split CQR requires a distinct calibration partition; not feasible with current windows",
        "final_evaluation": ranges["final_evaluation"],
        "test_period_used_for_selection": False,
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "sample_availability.json").write_text(json.dumps({
        "counts": counts,
        "calibration_feasible": counts["calibration"] > 0,
        "validation_and_calibration_reuse": False,
        "note": "The proposed calibration interval has only three calendar months and produces zero complete 10-month windows.",
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "calibration_audit.md").write_text(
        "# Calibration audit\n\n"
        "The current evaluator fits CQR corrections on validation predictions and then reports "
        "validation and test metrics. This is not a fully separated protocol because validation "
        "serves both model selection and probabilistic calibration. CQR is split-conformal in "
        "form, but temporal and site dependence mean nominal coverage is not independently guaranteed. "
        "The calibration sample is pooled across sites and leads within each SPEI scale, with lead-wise "
        "corrections; serial dependence and site dependence remain limitations.\n\n"
        "A distinct calibration partition with the 22-month embargo has zero complete windows, so "
        "the primary experiment must not claim calibrated coverage. No calibration reuse is introduced.\n",
        encoding="utf-8")
    (OUT / "validation_report.md").write_text(
        "# Phase 5.3 validation report\n\n"
        "Role assignment checks are implemented, but the current repaired chronology cannot provide "
        "a non-empty calibration partition. Validation remains suitable for model selection only; "
        "CQR calibration must be deferred or supported by an approved additional historical period. "
        "No model training or forecast generation was performed.\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.name != "sha256_manifest.json")
    (OUT / "sha256_manifest.json").write_text(
        json.dumps({path.name: sha256(path) for path in files}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
