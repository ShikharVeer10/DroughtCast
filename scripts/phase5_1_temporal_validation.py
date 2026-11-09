"""Generate additive Phase 5.1 temporal validation artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.protocol import SPLITS
from data.temporal_validation import TemporalRequirements, validate_samples

OUT = ROOT / "results" / "maharashtra" / "phase5" / "phase5_1_temporal_validation"
DATA = ROOT / "data" / "maharashtra" / "monthly_sites.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA, parse_dates=["date"])
    samples = []
    for site, group in frame.groupby("site"):
        group = group.sort_values("date")
        for start in range(len(group) - 6 - 4 + 1):
            dates = group["date"].iloc[start:start + 10].tolist()
            split = ("train" if dates[-1] <= pd.Timestamp(SPLITS["train_end"])
                     else "val" if dates[0] >= pd.Timestamp(SPLITS["val_start"]) and dates[-1] <= pd.Timestamp(SPLITS["val_end"])
                     else "test" if dates[0] >= pd.Timestamp(SPLITS["test_start"]) and dates[-1] <= pd.Timestamp(SPLITS["test_end"])
                     else None)
            if split:
                samples.append({"site": site, "split": split,
                                "input_dates": [str(d.date()) for d in dates[:6]],
                                "target_dates": [str(d.date()) for d in dates[6:]],
                                "target": "all"})
    requirements = TemporalRequirements(seq_len=6, forecast_steps=4)
    audit = validate_samples(samples, SPLITS, requirements)
    audit.update({"requirements": {"seq_len": 6, "forecast_steps": 4,
                                   "accumulation_windows": [3, 6, 12],
                                   "required_embargo_months": requirements.required_embargo_months},
                  "source": str(DATA.relative_to(ROOT)), "split_boundaries": SPLITS})
    (OUT / "temporal_leakage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    manifest = {"train": sum(1 for s in samples if s["split"] == "train"),
                "val": sum(1 for s in samples if s["split"] == "val"),
                "test": sum(1 for s in samples if s["split"] == "test")}
    pd.DataFrame([{"split": k, "sample_count": v} for k, v in manifest.items()]).to_csv(
        OUT / "sample_counts.csv", index=False)
    (OUT / "split_manifest.json").write_text(json.dumps({"boundaries": SPLITS, "counts": manifest}, indent=2), encoding="utf-8")
    (OUT / "embargo_derivation.md").write_text(
        "# Embargo derivation\n\n"
        "The loader uses six input months followed by four target months. "
        "Targets include SPEI-3, SPEI-6, and SPEI-12; the maximum accumulation dependency is 12 months. "
        "The conservative separation requirement is therefore 6 + 4 + 12 = 22 months. "
        "No additional calibration lag is added because calibration ends at the train boundary and "
        "index publication lags are already represented in input provenance.\n\n"
        "The configured gaps are 13 months (train to validation) and 13 months (validation to test), "
        "so both fail the derived requirement. Frozen boundaries are not changed in Phase 5.1.\n",
        encoding="utf-8")
    (OUT / "validation_report.md").write_text(
        f"# Temporal validation report\n\nOverall status: **{'PASS' if audit['pass'] else 'FAIL'}**.\n\n"
        f"Samples audited: **{audit['total_samples']}**. Required embargo: **22 months**. "
        "The existing frozen split boundaries do not meet that requirement.\n",
        encoding="utf-8")
    files = sorted(p for p in OUT.iterdir() if p.name != "sha256_manifest.json")
    (OUT / "sha256_manifest.json").write_text(
        json.dumps({p.name: sha256(p) for p in files}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
