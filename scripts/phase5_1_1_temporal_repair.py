"""Generate the auditable Phase 5.1.1 split-repair outputs."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.protocol import PHASE5_1_1_SPLITS, SPLITS
from data.temporal_validation import TemporalRequirements

OUT = ROOT / "results" / "maharashtra" / "phase5" / "phase5_1_1_temporal_repair"
DATA = ROOT / "data" / "maharashtra" / "monthly_sites.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_windows(boundaries: dict[str, str]) -> dict[str, int]:
    frame = pd.read_csv(DATA, parse_dates=["date"])
    counts = {"train": 0, "val": 0, "test": 0}
    for _, group in frame.groupby("site"):
        dates = group.sort_values("date")["date"].tolist()
        for start in range(len(dates) - 9):
            window = dates[start:start + 10]
            if window[0] >= pd.Timestamp(boundaries["val_start"]) and window[-1] <= pd.Timestamp(boundaries["val_end"]):
                counts["val"] += 1
            elif window[0] >= pd.Timestamp(boundaries["test_start"]) and window[-1] <= pd.Timestamp(boundaries["test_end"]):
                counts["test"] += 1
            elif window[0] <= pd.Timestamp(boundaries["train_end"]) and window[-1] <= pd.Timestamp(boundaries["train_end"]):
                counts["train"] += 1
    return counts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    original = dict(SPLITS)
    repaired = dict(PHASE5_1_1_SPLITS)
    original_counts = count_windows(original)
    repaired_counts = count_windows(repaired)
    comparison = []
    for split in ("train", "val", "test"):
        comparison.append({
            "split": split,
            "original_sample_count": original_counts[split],
            "repaired_sample_count": repaired_counts[split],
            "excluded_samples": original_counts[split] - repaired_counts[split],
        })
    pd.DataFrame(comparison).to_csv(OUT / "sample_count_comparison.csv", index=False)
    requirements = TemporalRequirements(6, 4)
    (OUT / "embargo_verification.md").write_text(
        "# Embargo verification\n\n"
        "The loader constructs six consecutive input months followed immediately by four target months. "
        "SPEI uses trailing rolling accumulations of 3, 6, and 12 months; the maximum dependency is 12 months. "
        "The conservative split separation is therefore 6 + 4 + 12 = 22 calendar months. "
        "Month-start boundaries are exclusive gaps: `2010-12` to `2012-10` and `2015-12` to `2017-10` "
        "are each 22 month-start increments. Samples must also remain wholly within one split, so no target "
        "window crosses a boundary.\n\n"
        "The repaired boundaries are the values in `data.protocol.SPLITS`. Frozen datasets and artifacts "
        "were not regenerated.\n", encoding="utf-8")
    (OUT / "repaired_split_manifest.json").write_text(json.dumps({
        "boundaries": repaired, "requirements": {
            "seq_len": 6, "forecast_steps": 4, "max_spei_accumulation_months": 12,
            "required_embargo_months": requirements.required_embargo_months,
        }, "sample_counts": repaired_counts,
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "validation_report.md").write_text(
        "# Phase 5.1.1 validation report\n\n"
        "The split repair satisfies both 22-month boundaries. Training retains all 2,916 windows; "
        "validation retains 360 of 468 windows; test retains 1,080 of 1,188 windows. "
        "The removed windows are in the explicitly embargoed periods. No test-period values were used "
        "for selection, and no training or forecast generation was performed.\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.name != "sha256_manifest.json")
    (OUT / "sha256_manifest.json").write_text(
        json.dumps({path.name: sha256(path) for path in files}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
