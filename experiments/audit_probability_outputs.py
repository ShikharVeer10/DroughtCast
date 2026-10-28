
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/inference_probabilities_seed_42.csv"),
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
    ordered = (frame.q10 <= frame.q50) & (frame.q50 <= frame.q90)
    invalid_probability = frame.calibrated_probability.isin([0.0, 1.0])
    failed = not ordered.all() or invalid_probability.any()
    grouped = frame.groupby(["scale", "lead"], sort=True)
    print(f"Audited {len(frame)} forecasts for {frame.site.nunique()} sites")
    print(f"Dataset hash: {actual_hash}")
    print("Scale/lead | unique q10/q50/q90 | unique raw P | unique calibrated P")
    identical_site_pairs = 0
    for (scale, lead), group in grouped:
        interior = group.loc[~group.calibrated_probability.isin([0.005, 0.995])]
        duplicate_share = (
            1 - interior.calibrated_probability.nunique() / len(interior)
            if len(interior) else 0.0
        )
        failed |= duplicate_share > 0.5
        counts = group.groupby(["q10", "q50", "q90"]).size()
        identical_site_pairs += int((counts * (counts - 1) // 2).sum())
        print(
            f"{scale}/{lead}: {group.q10.nunique()}/{group.q50.nunique()}/{group.q90.nunique()} | "
            f"{group.raw_probability.nunique()} | {group.calibrated_probability.nunique()} | "
            f"duplicate share={duplicate_share:.3f}"
        )
    print(f"Any exact calibrated 0/1: {bool(invalid_probability.any())}")
    print(f"Quantiles unordered: {int((~ordered).sum())}")
    print(f"Identical-site-pair count: {identical_site_pairs}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
