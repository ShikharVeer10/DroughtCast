"""Audit site-level probabilistic forecasts for identity, provenance, and calibration resolution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/inference_probabilities_seed_42.csv"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("results/probability_calibration/calibration_seed_42.json"),
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/monthly_sites.csv"))
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    required = {
        "site",
        "scale",
        "lead",
        "q10",
        "q50",
        "q90",
        "raw_probability",
        "calibrated_probability",
        "dataset_sha256",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.predictions}: {sorted(missing)}")
    actual_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    recorded = set(frame.dataset_sha256.astype(str))
    if recorded != {actual_hash}:
        raise ValueError(
            f"Dataset hash mismatch: CSV={recorded}, current={actual_hash}"
        )
    if frame.duplicated(["site", "scale", "lead"]).any():
        raise ValueError("Duplicate site/scale/lead forecast rows found")
    if not ((frame.q10 <= frame.q50) & (frame.q50 <= frame.q90)).all():
        raise ValueError("Quantile ordering violation found")
    grouped = frame.groupby(["scale", "lead"], sort=True)
    print(f"Audited {len(frame)} forecasts for {frame.site.nunique()} sites")
    print(f"Dataset hash: {actual_hash}")
    print("Scale/lead | unique q10/q50/q90 | unique raw P | unique calibrated P")
    for (scale, lead), group in grouped:
        print(
            f"{scale}/{lead}: {group.q10.nunique()}/{group.q50.nunique()}/{group.q90.nunique()} | "
            f"{group.raw_probability.nunique()} | {group.calibrated_probability.nunique()}"
        )
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    rows = [
        value
        for scale in calibration.get("scales", {}).values()
        for value in scale.values()
    ]
    worse = sum(
        float(row["calibrated_brier"]) > float(row["raw_brier"]) for row in rows
    )
    print(
        f"Calibration evaluation: {len(rows) - worse}/{len(rows)} scale/lead cells improved Brier score; "
        f"{worse}/{len(rows)} worsened it. ECE must be reported separately."
    )
    print(
        "Interpretation: repeated calibrated values are expected from piecewise-constant isotonic calibration; "
        "they do not imply duplicated q10/q50/q90 forecasts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
