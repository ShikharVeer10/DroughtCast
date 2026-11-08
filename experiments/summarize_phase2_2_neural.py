"""Add artifact-derived aggregate and site summaries to a Phase 2.2 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import t as student_t

BASELINE_ROOT = Path("results/maharashtra/phase2/baselines")
BASELINE_MODELS = (
    "persistence",
    "ridge",
    "random_forest",
    "xgboost",
    "seasonal_climatology",
)
SCALE_LABELS = {"spei_3": "SPEI-3", "spei_6": "SPEI-6", "spei_12": "SPEI-12"}


def append_result_summary(run_dir: Path) -> None:
    metadata = json.loads((run_dir / "neural_experiment_metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("test_evaluation_performed"):
        raise RuntimeError("Locked test results are not present")
    baseline_dir = BASELINE_ROOT / metadata["baseline_run_id"]
    neural = pd.read_csv(run_dir / "neural_metrics_test.csv")
    baseline_source = pd.read_csv(baseline_dir / "baseline_metrics.csv")
    baseline_filter = (baseline_source["feature_variant"] == "all_features") & baseline_source["model"].isin(BASELINE_MODELS)
    baseline = baseline_source.loc[baseline_filter & (baseline_source["site"] == "ALL")]
    baseline_site = baseline_source.loc[
        baseline_filter
        & (baseline_source["site"] != "ALL")
        & (baseline_source["model"] == "persistence")
        & (baseline_source["target_column"] == "spei_3")
        & baseline_source["lead"].isin(("T+3", "T+4")),
        ["site", "lead", "rmse"],
    ].rename(columns={"rmse": "persistence_rmse"})
    neural_all = neural.loc[neural["site"] == "ALL"]
    selected_profile = next(
        row for row in metadata["candidate_selection"] if row["profile"] == metadata["selected_profile"]
    )
    selected_epochs = {42: selected_profile["selected_epoch"]}
    selected_epochs.update(
        {int(row["seed"]): row["selected_epoch"] for row in selected_profile.get("multiseed_runs", [])}
    )
    primary_neural = neural.loc[
        (neural["site"] != "ALL")
        & (neural["target_column"] == "spei_3")
        & neural["lead"].isin(("T+3", "T+4")),
        ["seed", "site", "lead", "rmse"],
    ].rename(columns={"rmse": "neural_rmse"})
    primary = primary_neural.merge(
        baseline_site, on=["site", "lead"], validate="many_to_one"
    )
    primary["neural_mse"] = primary["neural_rmse"] ** 2
    primary["persistence_mse"] = primary["persistence_rmse"] ** 2
    primary_site = primary.groupby(["seed", "site"], as_index=False).agg(
        neural_mse=("neural_mse", "mean"),
        persistence_mse=("persistence_mse", "mean"),
    )
    primary_site["mse_skill"] = 1 - primary_site["neural_mse"] / primary_site["persistence_mse"]
    primary_seed = primary_site.groupby("seed")["mse_skill"].mean()
    primary_mean = float(primary_seed.mean())
    primary_sd = float(primary_seed.std(ddof=1))
    primary_half_width = float(student_t.ppf(0.975, df=len(primary_seed) - 1) * primary_sd / (len(primary_seed) ** 0.5))
    primary_by_site = primary_site.groupby("site")["mse_skill"].mean()

    lines = [
        "",
        "## Aggregate locked-test results and site variation",
        "",
        "The following scale summaries average the four leads. Neural metrics average the five locked seeds; classical baseline values are the already saved Phase 2.1 test results. This is descriptive comparison on the reused test period.",
        "",
        "Validation selected epoch by seed for the chosen profile: "
        + ", ".join(f"{seed}={epoch}" for seed, epoch in sorted(selected_epochs.items()))
        + ". The best primary validation score occurred at epoch 1 for every seed; patience stopped the runs later without a subsequent improvement.",
        "",
        f"The configured primary endpoint, equal-site SPEI-3 MSE skill versus persistence over T+3/T+4, was {primary_mean:+.4f} across the five seeds (seed-level 95% t interval {primary_mean - primary_half_width:+.4f} to {primary_mean + primary_half_width:+.4f}; seed SD {primary_sd:.4f}). Mean skill by site ranged from {primary_by_site.min():+.4f} to {primary_by_site.max():+.4f}. These intervals describe seed variation on the already-used benchmark and do not replace a fresh temporal evaluation.",
        "",
        "| Target | Neural q50 RMSE | Persistence | Ridge | Random forest | XGBoost | Seasonal climatology | q10-q90 coverage (nominal 80%) | Severe-category recall | SPEI <= -1.5 recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    baseline_mean = baseline.groupby(["target_column", "model"])["rmse"].mean()
    for scale, label in SCALE_LABELS.items():
        rows = neural_all.loc[neural_all["target_column"] == scale]
        values = {
            name: float(baseline_mean.loc[(scale, name)])
            for name in BASELINE_MODELS
        }
        lines.append(
            f"| {label} | {rows['rmse'].mean():.4f} | {values['persistence']:.4f} | "
            f"{values['ridge']:.4f} | {values['random_forest']:.4f} | {values['xgboost']:.4f} | "
            f"{values['seasonal_climatology']:.4f} | {rows['q10_q90_coverage'].mean():.4f} | "
            f"{rows['severe_drought_recall'].mean():.4f} | "
            f"{rows['severe_or_extreme_recall'].mean():.4f} |"
        )

    site_metrics = neural.loc[neural["site"] != "ALL"].groupby("site").agg(
        mean_rmse=("rmse", "mean"),
        mean_macro_f1=("macro_f1", "mean"),
        severe_or_extreme_recall=("severe_or_extreme_recall", "mean"),
    ).sort_values("mean_rmse", ascending=False)
    lines.extend(
        [
            "",
            "The neural model's mean q50 RMSE exceeds Ridge and persistence at all three target scales. It is close to seasonal climatology for SPEI-6 and lower than climatology for SPEI-12, but the latter is still substantially worse than persistence/Ridge. The nominal central 80% q10-q90 intervals under-cover on average; these quantiles are not calibrated drought probabilities.",
            "",
            "Worst and best site averages across scales, leads, and seeds (all 12 site rows remain in `neural_per_site_metrics.csv`):",
            "",
            "| Site | Mean RMSE | Mean macro F1 | Severe-or-extreme recall |",
            "|---|---:|---:|---:|",
        ]
    )
    selected_sites = pd.concat([site_metrics.head(3), site_metrics.tail(3)]).loc[lambda frame: ~frame.index.duplicated()]
    for site, row in selected_sites.iterrows():
        lines.append(
            f"| {site} | {row['mean_rmse']:.4f} | {row['mean_macro_f1']:.4f} | "
            f"{row['severe_or_extreme_recall']:.4f} |"
        )

    classes = pd.read_csv(run_dir / "neural_class_distribution.csv")
    classes = classes.loc[
        (classes["split"] == "test")
        & (classes["site"] == "ALL")
        & (classes["seed"] == 42)
    ]
    event_rows = []
    for scale, label in SCALE_LABELS.items():
        scale_rows = classes.loc[classes["target_column"] == scale]
        for lead in ("T+1", "T+2", "T+3", "T+4"):
            lead_rows = scale_rows.loc[scale_rows["lead"] == lead]
            positive_count = int(
                lead_rows.loc[
                    lead_rows["category"].isin(("severe_drought", "extreme_drought")),
                    "count",
                ].sum()
            )
            event_rows.append((label, lead, positive_count))
    lines.extend(
        [
            "",
            "Severe-or-extreme observed target-window counts (category severe plus extreme) and neural detection:",
            "",
            "| Target | Lead | Observed cases with SPEI <= -1.5 | Neural true positives, seed 42 |",
            "|---|---|---:|---:|",
        ]
    )
    for label, lead, positive_count in event_rows:
        row = neural_all.loc[
            (neural_all["target_column"] == label.lower().replace("-", "_"))
            & (neural_all["lead"] == lead)
            & (neural_all["seed"] == 42)
        ]
        true_positives = (
            round(row["severe_or_extreme_recall"].iloc[0] * positive_count)
            if not row.empty
            else 0
        )
        lines.append(f"| {label} | {lead} | {positive_count} | {true_positives} |")
    lines.extend(
        [
            "",
            "Artifact limitation: the confusion-matrix JSON key omits the seed, so its selected-profile validation and locked-test matrices represent seed 2026 only (the final seed processed). Per-seed regression/category scores and class distributions are retained in the CSV files; full per-seed multiclass matrices were not retained and are not reconstructed by rerunning the test set.",
        ]
    )

    report_path = run_dir / "neural_experiment_report.md"
    existing = report_path.read_text(encoding="utf-8")
    existing = existing.replace(
        "test labels were not read until after the evaluation manifest was frozen.",
        "the loader parses the source CSV but filters later dates before quality checks, normalization, and window construction; no test targets were used for fitting or model selection, and test windows were first materialized after the freeze.",
    )
    existing = existing.replace(
        "the test split was not read until freeze.",
        "test rows were excluded before preprocessing and window construction, and were used only for the locked evaluation after freeze.",
    )
    marker = "\n## Aggregate locked-test results and site variation\n"
    if marker in existing:
        existing = existing.split(marker, maxsplit=1)[0]
    report_path.write_text(existing.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    append_result_summary(args.run_dir)


if __name__ == "__main__":
    main()
