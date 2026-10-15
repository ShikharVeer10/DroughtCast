"""Aggregate per-seed held-out metrics into machine-readable CSV summaries."""
import json
import sys
from pathlib import Path

import pandas as pd


def main(seeds):
    rows = []
    for seed in seeds:
        path = Path(f"results/seeds/metrics_seed_{seed}.json")
        report = json.loads(path.read_text(encoding="utf-8"))
        for method, scales in report["methods"].items():
            for scale, leads in scales.items():
                for lead, metrics in leads.items():
                    if not lead.startswith("T+"):
                        continue
                    rows.append({"seed": seed, "method": method, "scale": scale, "lead": lead, **metrics})
    frame = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    frame.to_csv("results/per_seed.csv", index=False)
    summary = frame.groupby(["method", "scale", "lead"], as_index=False).agg(
        n_seeds=("seed", "nunique"), rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"), nse_mean=("nse", "mean"),
        nse_std=("nse", "std"), kge_mean=("kge_2012", "mean"), kge_std=("kge_2012", "std"),
        skill_mean=("skill_vs_persistence", "mean"), skill_std=("skill_vs_persistence", "std"))
    summary.to_csv("results/summary.csv", index=False)
    paper_rows = summary[summary.method.isin(["droughtcast", "persistence", "ridge", "hist_gradient_boosting"])]
    lines = ["| Method | Scale | Lead | RMSE mean ± std | Skill vs persistence |", "|---|---|---|---:|---:|"]
    for row in paper_rows.itertuples(index=False):
        skill = "—" if pd.isna(row.skill_mean) else f"{row.skill_mean:.3f} ± {row.skill_std:.3f}"
        lines.append(f"| {row.method} | {row.scale.upper()} | {row.lead} | {row.rmse_mean:.3f} ± {row.rmse_std:.3f} | {skill} |")
    Path("results/results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]])
