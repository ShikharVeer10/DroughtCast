"""Aggregate validation-only event calibration across predetermined seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/probability_calibration")
    parser.add_argument("--output", default="results/probability_calibration/summary.csv")
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.input_dir).glob("calibration_seed_*.json")):
        seed = int(path.stem.rsplit("_", 1)[1])
        report = json.loads(path.read_text(encoding="utf-8"))
        for scale, leads in report["scales"].items():
            for lead, values in leads.items():
                rows.append({"seed": seed, "scale": scale, "lead": lead, **{
                    key: values[key] for key in ("raw_brier", "calibrated_brier", "raw_ece", "calibrated_ece")
                }})
    if not rows:
        raise FileNotFoundError("No calibration_seed_*.json reports found")
    frame = pd.DataFrame(rows)
    summary = (frame.groupby(["scale", "lead"], as_index=False)
               .agg(seeds=("seed", "nunique"),
                    raw_brier_mean=("raw_brier", "mean"),
                    raw_brier_std=("raw_brier", "std"),
                    calibrated_brier_mean=("calibrated_brier", "mean"),
                    calibrated_brier_std=("calibrated_brier", "std"),
                    raw_ece_mean=("raw_ece", "mean"),
                    calibrated_ece_mean=("calibrated_ece", "mean")))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    frame.to_csv(output.with_name("per_seed.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
