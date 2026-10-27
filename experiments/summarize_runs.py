"""Summarize multi-seed evaluations with Mean +/- Standard Deviation and significance metrics."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data.loader import validate_dataset_protocol

SCALES = ["spei_3", "spei_6", "spei_12"]
LEADS = ["T+1", "T+2", "T+3", "T+4"]


def main():
    seeds = sys.argv[1:] if len(sys.argv) > 1 else ["42", "101", "2024", "2025", "2026"]
    results_dir = Path("results/seeds")
    reports = []
    for s in seeds:
        p = results_dir / f"metrics_seed_{s}.json"
        if p.exists():
            reports.append((s, json.loads(p.read_text(encoding="utf-8"))))
        else:
            print(f"Warning: {p} not found")

    if not reports:
        print("No seed reports found.")
        return

    dataset_protocol = validate_dataset_protocol()
    expected_protocol = {
        "schema_version": dataset_protocol["schema_version"],
        "index_alignment": dataset_protocol["index_alignment"],
    }
    stale_reports = [
        str(report["seed"])
        for _, report in reports
        if report.get("dataset_protocol") != expected_protocol
    ]
    if stale_reports:
        raise RuntimeError(
            "Seed reports lack the current dataset protocol and cannot be summarized: "
            + ", ".join(stale_reports)
            + ". Re-run evaluation for each seed after rebuilding the dataset."
        )

    methods = list(reports[0][1]["methods"].keys())
    records = []

    for seed_str, rep in reports:
        for m in methods:
            if m not in rep["methods"]:
                continue
            for scale in SCALES:
                if scale not in rep["methods"][m]:
                    continue
                for lead in LEADS:
                    if lead not in rep["methods"][m][scale]:
                        continue
                    info = rep["methods"][m][scale][lead]
                    row = {
                        "seed": int(seed_str),
                        "method": m,
                        "scale": scale,
                        "lead": lead,
                        "rmse": info.get("rmse", np.nan),
                        "mae": info.get("mae", np.nan),
                        "nse": info.get("nse", np.nan),
                        "kge_2012": info.get("kge_2012", np.nan),
                        "skill_vs_persistence": info.get(
                            "skill_vs_persistence", np.nan
                        ),
                        "csi_mod": info.get("drought_event_moderate", {}).get(
                            "csi", np.nan
                        ),
                        "pod_mod": info.get("drought_event_moderate", {}).get(
                            "pod", np.nan
                        ),
                        "far_mod": info.get("drought_event_moderate", {}).get(
                            "far", np.nan
                        ),
                    }
                    records.append(row)

    df = pd.DataFrame(records)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_seed.csv", index=False)

    # Compute Mean and Standard Deviation across seeds
    summary_rows = []
    grouped = df.groupby(["method", "scale", "lead"])
    for (m, s, l), g in grouped:
        summary_rows.append(
            {
                "method": m,
                "scale": s,
                "lead": l,
                "n_seeds": len(g),
                "rmse_mean": g["rmse"].mean(),
                "rmse_std": g["rmse"].std(ddof=1) if len(g) > 1 else 0.0,
                "mae_mean": g["mae"].mean(),
                "mae_std": g["mae"].std(ddof=1) if len(g) > 1 else 0.0,
                "nse_mean": g["nse"].mean(),
                "nse_std": g["nse"].std(ddof=1) if len(g) > 1 else 0.0,
                "kge_mean": g["kge_2012"].mean(),
                "kge_std": g["kge_2012"].std(ddof=1) if len(g) > 1 else 0.0,
                "skill_mean": g["skill_vs_persistence"].mean(),
                "skill_std": g["skill_vs_persistence"].std(ddof=1)
                if len(g) > 1
                else 0.0,
                "csi_mean": g["csi_mod"].mean(),
                "pod_mean": g["pod_mod"].mean(),
                "far_mean": g["far_mod"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (Path("results") / "run_manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "dataset_schema_version": dataset_protocol["schema_version"],
                "index_alignment": dataset_protocol["index_alignment"],
                "seeds": [int(seed) for seed, _ in reports],
                "methods": methods,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n================ MULTI-SEED BENCHMARK SUMMARY (MEAN +/- STD) ================"
    )
    pivot = summary_df.pivot(
        index=["scale", "lead"],
        columns="method",
        values=["rmse_mean", "kge_mean", "skill_mean"],
    )
    print(pivot.to_string())
    print(
        f"\nSaved summary to {out_dir / 'summary.csv'} and {out_dir / 'per_seed.csv'}"
    )


if __name__ == "__main__":
    main()
