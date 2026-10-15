"""Create long-form mean/std tables from experiment ablation JSON files."""
import json
import sys
from pathlib import Path

import pandas as pd


def main(root: Path):
    rows = []
    for variant_dir in root.iterdir():
        if not variant_dir.is_dir() or not (variant_dir / "config.yaml").exists():
            continue
        for report_path in variant_dir.glob("metrics_seed_*.json"):
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for method, scales in report["methods"].items():
                for scale, leads in scales.items():
                    for lead, metrics in leads.items():
                        if lead.startswith("T+"):
                            rows.append({"variant": variant_dir.name, "seed": report["seed"],
                                         "method": method, "scale": scale, "lead": lead, **metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "per_seed.csv", index=False)
    if frame.empty:
        print("No ablation reports found")
        return
    summary = frame.groupby(["variant", "method", "scale", "lead"], as_index=False).agg(
        n_seeds=("seed", "nunique"), rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"), skill_mean=("skill_vs_persistence", "mean"),
        skill_std=("skill_vs_persistence", "std"))
    summary.to_csv(root / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/ablations"))
