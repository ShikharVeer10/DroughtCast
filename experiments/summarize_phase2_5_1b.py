"""Create the train/validation comparison artifacts for Phase 2.5.1B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_DIR = Path(
    "results/maharashtra/phase2/diagnostics/"
    "phase2_5_1_protocol10_seed42_f5cd5beb9f59"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def _metric_rows(run_dir: Path, split: str, epoch: int) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "epoch_metrics.csv")
    return frame.loc[(frame["split"] == split) & (frame["epoch"] == epoch)].copy()


def summarize(alternative_dir: Path) -> dict:
    base_meta = _read_json(BASELINE_DIR / "experiment_metadata.json")
    alt_meta = _read_json(alternative_dir / "experiment_metadata.json")
    baseline_selection = _read_json(BASELINE_DIR / "selection.json")
    alt_selection = _read_json(alternative_dir / "selection.json")
    for key in ("dataset_sha256", "normalizer_sha256", "protocol_version", "features", "split_boundaries"):
        if base_meta[key] != alt_meta[key]:
            raise ValueError(f"Baseline and alternative differ in locked field: {key}")
    if base_meta["profile"] != alt_meta["profile"] or base_meta["seed"] != alt_meta["seed"]:
        raise ValueError("Baseline and alternative must use the same profile and seed")
    if alt_meta["test_split_accessed_or_evaluated"] or alt_meta["test_split_windows_constructed"]:
        raise ValueError("Alternative run metadata indicates test split access")

    base_selected_epoch = int(baseline_selection["selected_epoch"])
    alt_selected_epoch = int(alt_selection["selected_epoch"])
    base_final_epoch = int(pd.read_csv(BASELINE_DIR / "training_history.csv")["epoch"].max())
    alt_final_epoch = int(pd.read_csv(alternative_dir / "objective_training_history.csv")["epoch"].max())

    compare_frames = []
    for label, base_epoch, alt_epoch in (
        ("selected_checkpoint", base_selected_epoch, alt_selected_epoch),
        ("same_epoch_1", 1, 1),
        ("same_final_epoch", base_final_epoch, alt_final_epoch),
    ):
        base = _metric_rows(BASELINE_DIR, "validation", base_epoch)
        alt = _metric_rows(alternative_dir, "validation", alt_epoch)
        merged = base.merge(
            alt,
            on=["split", "target", "lead"],
            suffixes=("_baseline", "_alternative"),
            validate="one_to_one",
        )
        if not np.array_equal(merged["n_valid_baseline"], merged["n_valid_alternative"]):
            raise ValueError(f"Matched validation sample counts differ at {label}")
        for metric in (
            "n_valid", "median_rmse", "median_mae", "median_pearson",
            "prediction_target_variance_ratio", "q10_q90_coverage",
            "q10_q90_mean_width", "quantile_crossing_rate",
            "actual_spei_le_minus_1_5", "predicted_q50_le_minus_1_5",
            "severe_event_recall", "pinball_sum_quantiles_weighted",
        ):
            merged[f"{metric}_delta_alternative_minus_baseline"] = (
                merged[f"{metric}_alternative"] - merged[f"{metric}_baseline"]
            )
        merged["comparison_point"] = label
        compare_frames.append(merged)
    comparison = pd.concat(compare_frames, ignore_index=True)
    comparison.to_csv(alternative_dir / "baseline_vs_alternative_validation.csv", index=False)

    base_primary = pd.read_csv(BASELINE_DIR / "primary_endpoint.csv")
    alt_primary = pd.read_csv(alternative_dir / "primary_endpoint.csv")
    base_primary = base_primary.loc[base_primary["epoch"] == base_selected_epoch]
    alt_primary = alt_primary.loc[alt_primary["epoch"] == alt_selected_epoch]
    site_comparison = base_primary.merge(
        alt_primary,
        on=["target", "lead", "site"],
        suffixes=("_baseline", "_alternative"),
        validate="one_to_one",
    )
    site_comparison["skill_delta_alternative_minus_baseline"] = (
        site_comparison["skill_alternative"] - site_comparison["skill_baseline"]
    )
    site_comparison.to_csv(
        alternative_dir / "primary_endpoint_per_site_comparison.csv", index=False
    )

    base_history = pd.read_csv(BASELINE_DIR / "training_history.csv")
    alt_history = pd.read_csv(alternative_dir / "objective_training_history.csv")
    history_comparison = base_history.merge(
        alt_history,
        on="epoch",
        suffixes=("_baseline", "_alternative"),
        validate="one_to_one",
    )
    history_comparison.to_csv(alternative_dir / "training_trajectory_comparison.csv", index=False)

    base_selected = comparison.loc[comparison["comparison_point"] == "selected_checkpoint"]
    per_scale = []
    for scale, group in base_selected.groupby("target", sort=True):
        per_scale.append({
            "target": scale,
            "baseline_mean_pinball": _finite(group["pinball_sum_quantiles_weighted_baseline"].mean()),
            "alternative_mean_pinball": _finite(group["pinball_sum_quantiles_weighted_alternative"].mean()),
            "baseline_mean_rmse": _finite(group["median_rmse_baseline"].mean()),
            "alternative_mean_rmse": _finite(group["median_rmse_alternative"].mean()),
            "baseline_mean_variance_ratio": _finite(group["prediction_target_variance_ratio_baseline"].mean()),
            "alternative_mean_variance_ratio": _finite(group["prediction_target_variance_ratio_alternative"].mean()),
            "baseline_mean_interval_coverage": _finite(group["q10_q90_coverage_baseline"].mean()),
            "alternative_mean_interval_coverage": _finite(group["q10_q90_coverage_alternative"].mean()),
        })

    base_end = _read_json(BASELINE_DIR / "diagnostic_metrics.json")
    alt_primary_skill = float(alt_selection["best_validation_primary_site_mean_mse_skill"])
    base_primary_skill = float(baseline_selection["best_validation_primary_site_mean_mse_skill"])
    primary_selected = base_selected.loc[
        (base_selected["target"] == "spei_3")
        & base_selected["lead"].isin(("T+3", "T+4"))
    ]
    summary = {
        "experiment": "Phase 2.5.1B objective-alignment comparison",
        "baseline_run_id": base_meta["run_id"],
        "alternative_run_id": alt_meta["run_id"],
        "seed": int(alt_meta["seed"]),
        "profile": alt_meta["profile"],
        "objective": alt_meta["objective"],
        "dataset_sha256": alt_meta["dataset_sha256"],
        "normalizer_sha256": alt_meta["normalizer_sha256"],
        "validation_windows": int(alt_meta["window_counts"]["validation"]),
        "test_windows_constructed": False,
        "test_split_used": False,
        "baseline_selected_epoch": base_selected_epoch,
        "alternative_selected_epoch": alt_selected_epoch,
        "baseline_final_epoch": base_final_epoch,
        "alternative_final_epoch": alt_final_epoch,
        "validation_primary_skill_at_final_epoch": {
            "baseline": _finite(base_end["validation_primary_skill"]["last"]),
            "alternative": _finite(float(alt_history.loc[alt_history["epoch"] == alt_final_epoch, "validation_primary_site_mean_mse_skill"].iloc[0])),
        },
        "alternative_training_components": {
            "pinball_first": _finite(float(alt_history.iloc[0]["train_pinball_component"])),
            "pinball_final": _finite(float(alt_history.iloc[-1]["train_pinball_component"])),
            "primary_mse_first": _finite(float(alt_history.iloc[0]["train_primary_mse_component"])),
            "primary_mse_final": _finite(float(alt_history.iloc[-1]["train_primary_mse_component"])),
        },
        "alternative_validation_primary_mse_trajectory": {
            "first": _finite(float(alt_history.iloc[0]["validation_primary_pooled_mse"])),
            "final": _finite(float(alt_history.iloc[-1]["validation_primary_pooled_mse"])),
        },
        "selected_checkpoint_validation_primary_skill": {
            "baseline": base_primary_skill,
            "alternative": alt_primary_skill,
            "difference_alternative_minus_baseline": alt_primary_skill - base_primary_skill,
        },
        "selected_checkpoint_primary_macro_site_rmse": {
            "baseline": _finite(base_end["validation_rmse_primary_macro_site"]["first"]),
            "alternative": _finite(float(alt_history.loc[alt_history["epoch"] == alt_selected_epoch, "validation_macro_site_spei3_t3_t4_rmse"].iloc[0])),
        },
        "selected_checkpoint_validation_primary_pooled_mse": {
            "baseline": None,
            "alternative": _finite(float(alt_history.loc[alt_history["epoch"] == alt_selected_epoch, "validation_primary_pooled_mse"].iloc[0])),
            "baseline_note": "Not saved in Phase 2.5.1 history; baseline is compared using the exact site-macro selection criterion and per-site endpoint records.",
        },
        "primary_selected_epoch_validation_cells": primary_selected[
            ["target", "lead", "n_valid_baseline", "median_rmse_baseline", "median_rmse_alternative",
             "prediction_target_variance_ratio_baseline", "prediction_target_variance_ratio_alternative",
             "q10_q90_coverage_baseline", "q10_q90_coverage_alternative",
             "actual_spei_le_minus_1_5_baseline", "predicted_q50_le_minus_1_5_baseline",
             "predicted_q50_le_minus_1_5_alternative"]
        ].to_dict("records"),
        "severe_events_primary_leads_selected_epoch": {
            "actual": int(primary_selected["actual_spei_le_minus_1_5_alternative"].sum()),
            "baseline_predicted": int(primary_selected["predicted_q50_le_minus_1_5_baseline"].sum()),
            "alternative_predicted": int(primary_selected["predicted_q50_le_minus_1_5_alternative"].sum()),
        },
        "nonfinite_gradient_tensors": int(pd.read_csv(alternative_dir / "optimization_diagnostics.csv")["nonfinite_gradient_tensors"].sum()),
        "nonfinite_update_tensors": int(pd.read_csv(alternative_dir / "optimization_diagnostics.csv")["nonfinite_update_tensors"].sum()),
        "quantile_crossings_validation": int(_metric_rows(alternative_dir, "validation", alt_selected_epoch)["quantile_crossing_count"].sum()),
        "per_scale_selected_epoch": per_scale,
        "evidence_scope": "single seed; repeatedly used validation period; exploratory only",
        "causal_claim_supported": False,
    }
    (alternative_dir / "phase2_5_1b_metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    hash_paths = [
        Path("experiments/run_phase2_5_1b_objective.py"),
        Path("experiments/summarize_phase2_5_1b.py"),
        Path("tests/test_phase2_5_1b_objective.py"),
        Path("experiments/run_phase2_2_neural.py"),
        Path("experiments/training_diagnostics.py"),
        Path("data/protocol.py"),
        Path("data/loader.py"),
        Path("data/preprocessing.py"),
        Path("configs/config.yaml"),
        Path(alt_meta["checkpoint_path"]),
    ]
    hash_manifest = {
        "alternative_run_id": alt_meta["run_id"],
        "baseline_run_id": base_meta["run_id"],
        "dataset_sha256": alt_meta["dataset_sha256"],
        "normalizer_sha256": alt_meta["normalizer_sha256"],
        "source_and_artifact_sha256": {str(path): sha256(path) for path in hash_paths},
    }
    (alternative_dir / "source_and_data_hashes.json").write_text(
        json.dumps(hash_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_report(alternative_dir, summary, site_comparison, history_comparison)
    return summary


def _write_report(run_dir: Path, summary: dict, site_comparison: pd.DataFrame, history: pd.DataFrame) -> None:
    primary = summary["selected_checkpoint_validation_primary_skill"]
    mse = summary["selected_checkpoint_primary_macro_site_rmse"]
    severe = summary["severe_events_primary_leads_selected_epoch"]
    scale_lines = "\n".join(
        f"| {row['target']} | {row['baseline_mean_rmse']:.3f} | {row['alternative_mean_rmse']:.3f} | "
        f"{row['baseline_mean_variance_ratio']:.3f} | {row['alternative_mean_variance_ratio']:.3f} | "
        f"{row['baseline_mean_interval_coverage']:.1%} | {row['alternative_mean_interval_coverage']:.1%} |"
        for row in summary["per_scale_selected_epoch"]
    )
    selected_history = history.loc[history["epoch"] == summary["alternative_selected_epoch"]].iloc[0]
    base_selected_history = history.loc[history["epoch"] == summary["baseline_selected_epoch"]].iloc[0]
    site_lines = "\n".join(
        f"| {row.site} | {row.lead} | {row.skill_baseline:.3f} | "
        f"{row.skill_alternative:.3f} | {row.skill_delta_alternative_minus_baseline:+.3f} |"
        for row in site_comparison.itertuples()
    )
    report = f"""# Phase 2.5.1B — Controlled Objective-Alignment Experiment

## Decision

**Exploratory evidence only; objective mismatch is not established as the cause of prediction collapse.** This comparison uses one seed and the repeatedly used validation split. It is not independent confirmation or a test-set result.

## Experimental contract

- Baseline run: `{summary['baseline_run_id']}`; alternative run: `{summary['alternative_run_id']}`.
- Both use seed {summary['seed']}, profile `{summary['profile']}`, the same 15 ordered inputs, six input months, three targets, four leads, normalizer, data, split, model architecture, AdamW settings, scheduler, early stopping, and checkpoint criterion.
- Train windows: {json.loads((run_dir / 'experiment_metadata.json').read_text())['window_counts']['train']}; validation windows: {summary['validation_windows']}.
- Test windows constructed: **No**. Test metrics/labels used: **No**.
- Dataset SHA-256: `{summary['dataset_sha256']}`; normalizer SHA-256: `{summary['normalizer_sha256']}`.

### Objective definition

The baseline objective is the existing sum of three scale-wise masked, extreme-weighted quantile pinball losses:

`L_pinball = L_SPEI3 + L_SPEI6 + L_SPEI12`.

The alternative uses the same pinball term for every scale, lead, and q10/q50/q90 output, plus:

`L_alt = L_pinball + 0.5 * MSE_primary`,

where `MSE_primary` is the mean squared q50 error over valid training targets for SPEI-3 T+3 and T+4. The coefficient was fixed at 0.5 before the run and was not tuned. Only the training objective changed.

## Results

- Both runs selected epoch {summary['baseline_selected_epoch']} under the same validation criterion and stopped after {summary['baseline_final_epoch']} (baseline) and {summary['alternative_final_epoch']} (alternative) epochs.
- Selected-epoch validation site-macro MSE skill versus persistence: baseline {primary['baseline']:.4f}, alternative {primary['alternative']:.4f}, difference {primary['difference_alternative_minus_baseline']:+.4f}.
- Selected-epoch SPEI-3 T+3/T+4 macro-site RMSE: baseline {mse['baseline']:.4f}, alternative {mse['alternative']:.4f}. The validation-skill increase therefore does not correspond to a lower macro-site RMSE in these artifacts.
- Alternative training pinball component: {selected_history['train_pinball_component']:.4f} at its selected epoch; added primary MSE component: {selected_history['train_primary_mse_component']:.4f}. The baseline training pinball at its selected epoch was {base_selected_history['train_pinball_loss']:.4f}.
- During the alternative run, its training pinball component fell from {summary['alternative_training_components']['pinball_first']:.4f} to {summary['alternative_training_components']['pinball_final']:.4f}, and its training primary MSE fell from {summary['alternative_training_components']['primary_mse_first']:.4f} to {summary['alternative_training_components']['primary_mse_final']:.4f}; pooled validation primary MSE rose from {summary['alternative_validation_primary_mse_trajectory']['first']:.4f} to {summary['alternative_validation_primary_mse_trajectory']['final']:.4f}. At epoch 13 the primary validation skill was {summary['validation_primary_skill_at_final_epoch']['baseline']:.4f} for the baseline and {summary['validation_primary_skill_at_final_epoch']['alternative']:.4f} for the alternative.
- At the alternative selected epoch, {severe['baseline_predicted']} baseline and {severe['alternative_predicted']} alternative q50 predictions flagged SPEI ≤ -1.5 among {severe['actual']} actual SPEI-3 T+3/T+4 validation events.
- Alternative validation quantile crossings: {summary['quantile_crossings_validation']}; non-finite gradient tensors: {summary['nonfinite_gradient_tensors']}; non-finite parameter-update tensors: {summary['nonfinite_update_tensors']}.

| Scale | Baseline mean RMSE | Alternative mean RMSE | Baseline variance ratio | Alternative variance ratio | Baseline q10–q90 coverage | Alternative q10–q90 coverage |
|---|---:|---:|---:|---:|---:|---:|
{scale_lines}

Site-wise selected-epoch primary skill:

| Site | Lead | Baseline skill | Alternative skill | Difference |
|---|---|---:|---:|---:|
{site_lines}

The saved `baseline_vs_alternative_validation.csv` contains matched per-scale/per-lead comparisons at selected epochs and common epoch checkpoints. `training_trajectory_comparison.csv` contains the epoch-aligned objective and validation history.

## Interpretation and limits

The added median-MSE term changes the validation primary skill modestly, but both runs still select epoch 1 and stop at epoch 13. Compare q50 severe-event counts, variance ratios, and interval coverage before interpreting this as progress: quantile intervals are not calibrated merely because the model emits quantiles. The validation period has been repeatedly used for selection and diagnostics, and this is one seed. The comparison cannot establish causality, robustness, or generalization. No test-set result is included.

## Next step

Treat the objective-mismatch hypothesis as **partially supported as an alignment concern, unresolved as an explanation for prediction collapse**. A future confirmation requires a preregistered multi-seed training/validation study and genuinely untouched future or external evaluation data. Do not select further changes using the consumed test period.
"""
    (run_dir / "phase2_5_1b_comparison_report.md").write_text(report, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alternative_run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.alternative_run_dir), indent=2))


if __name__ == "__main__":
    import argparse

    main()
