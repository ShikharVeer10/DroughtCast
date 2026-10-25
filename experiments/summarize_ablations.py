"""Compile 5-row systematic ablation table quantifying exact metric degradation."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCALES = ["spei_3", "spei_6", "spei_12"]


def summarize_ablations(root_dir: Path):
    variants = sorted([p.name for p in root_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if not variants:
        print("No ablation variant directories found.")
        return

    table_rows = []
    for var in variants:
        var_dir = root_dir / var
        reports = list(var_dir.glob("metrics_seed_*.json"))
        if not reports:
            continue

        rmse_list, mae_list, kge_list, nse_list, skill_list = [], [], [], [], []
        for rep_path in reports:
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            dcast = rep.get("methods", {}).get("droughtcast", {})
            for scale in SCALES:
                for lead in ["T+1", "T+2", "T+3", "T+4"]:
                    if scale in dcast and lead in dcast[scale]:
                        info = dcast[scale][lead]
                        rmse_list.append(info.get("rmse", np.nan))
                        mae_list.append(info.get("mae", np.nan))
                        kge_list.append(info.get("kge_2012", np.nan))
                        nse_list.append(info.get("nse", np.nan))
                        skill_list.append(info.get("skill_vs_persistence", np.nan))

        table_rows.append({
            "variant": var,
            "n_seeds": len(reports),
            "rmse_mean": float(np.nanmean(rmse_list)),
            "rmse_std": float(np.nanstd(rmse_list)),
            "mae_mean": float(np.nanmean(mae_list)),
            "mae_std": float(np.nanstd(mae_list)),
            "kge_mean": float(np.nanmean(kge_list)),
            "nse_mean": float(np.nanmean(nse_list)),
            "skill_mean": float(np.nanmean(skill_list)),
        })

    df = pd.DataFrame(table_rows)
    if "full_model" in df["variant"].values:
        baseline_rmse = df.loc[df["variant"] == "full_model", "rmse_mean"].values[0]
        baseline_kge = df.loc[df["variant"] == "full_model", "kge_mean"].values[0]
        df["delta_rmse"] = df["rmse_mean"] - baseline_rmse
        df["delta_kge"] = df["kge_mean"] - baseline_kge

    print("\n================ SYSTEMATIC ABLATION TABLE (OVERALL 4-STEP POOLED) ================")
    print(df.to_string(index=False))
    df.to_csv(root_dir / "ablation_summary.csv", index=False)
    # Also write LaTeX table for direct inclusion in manuscript
    latex_path = root_dir / "ablation_table.tex"
    df.to_latex(latex_path, index=False, float_format="%.4f")
    print(f"\nSaved LaTeX ablation table to {latex_path}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/ablations")
    summarize_ablations(path)
