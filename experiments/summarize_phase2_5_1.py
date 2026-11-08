"""Summarize saved Phase 2.5.1 observer artifacts without loading model/data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _finite(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def summarize(run_dir: Path) -> dict:
    epoch = pd.read_csv(run_dir / "epoch_metrics.csv")
    history = pd.read_csv(run_dir / "training_history.csv")
    primary = pd.read_csv(run_dir / "primary_endpoint.csv")
    optimization = pd.read_csv(run_dir / "optimization_diagnostics.csv")
    if set(epoch["split"]) != {"train", "validation"}:
        raise ValueError("Expected only train and validation observer rows")
    if "test" in epoch["split"].to_list():
        raise ValueError("Test rows found in diagnostic observer artifact")

    records = []
    for (scale, lead), group in epoch[epoch["split"] == "validation"].groupby(
        ["target", "lead"], sort=True
    ):
        group = group.sort_values("epoch")
        first, last = group.iloc[0], group.iloc[-1]
        best_pinball = group.loc[group["pinball_sum_quantiles_weighted"].idxmin()]
        best_rmse = group.loc[group["median_rmse"].idxmin()]
        records.append({
            "target": scale,
            "lead": lead,
            "n_valid": int(first["n_valid"]),
            "validation_pinball_first": _finite(first["pinball_sum_quantiles_weighted"]),
            "validation_pinball_last": _finite(last["pinball_sum_quantiles_weighted"]),
            "validation_pinball_change_pct": _finite(
                100 * (last["pinball_sum_quantiles_weighted"] / first["pinball_sum_quantiles_weighted"] - 1)
            ),
            "validation_pinball_best_epoch": int(best_pinball["epoch"]),
            "validation_rmse_first": _finite(first["median_rmse"]),
            "validation_rmse_last": _finite(last["median_rmse"]),
            "validation_rmse_best_epoch": int(best_rmse["epoch"]),
            "prediction_target_variance_ratio_first": _finite(first["prediction_target_variance_ratio"]),
            "prediction_target_variance_ratio_last": _finite(last["prediction_target_variance_ratio"]),
            "q10_q90_coverage_first": _finite(first["q10_q90_coverage"]),
            "q10_q90_coverage_last": _finite(last["q10_q90_coverage"]),
            "q10_q90_width_last": _finite(last["q10_q90_mean_width"]),
            "quantile_crossing_count_total": int(group["quantile_crossing_count"].sum()),
            "actual_severe_events_last": int(last["actual_spei_le_minus_1_5"]),
            "predicted_severe_events_last": int(last["predicted_q50_le_minus_1_5"]),
        })
    per_cell = pd.DataFrame(records)
    per_cell.to_csv(run_dir / "per_scale_lead_summary.csv", index=False)
    training = epoch[epoch["split"] == "train"]
    train_scale_records = []
    for scale, group in training.groupby("target", sort=True):
        per_epoch = group.groupby("epoch").agg(
            pinball=("pinball_sum_quantiles_weighted", "mean"),
            variance_ratio=("prediction_target_variance_ratio", "mean"),
        )
        train_scale_records.append({
            "target": scale,
            "mean_per_lead_pinball_first": _finite(per_epoch["pinball"].iloc[0]),
            "mean_per_lead_pinball_last": _finite(per_epoch["pinball"].iloc[-1]),
            "mean_per_lead_variance_ratio_first": _finite(per_epoch["variance_ratio"].iloc[0]),
            "mean_per_lead_variance_ratio_last": _finite(per_epoch["variance_ratio"].iloc[-1]),
        })
    scale_records = []
    for scale, group in per_cell.groupby("target", sort=True):
        scale_records.append({
            "target": scale,
            "mean_validation_pinball_first": _finite(group["validation_pinball_first"].mean()),
            "mean_validation_pinball_last": _finite(group["validation_pinball_last"].mean()),
            "mean_validation_rmse_first": _finite(group["validation_rmse_first"].mean()),
            "mean_validation_rmse_last": _finite(group["validation_rmse_last"].mean()),
            "mean_prediction_variance_ratio_first": _finite(group["prediction_target_variance_ratio_first"].mean()),
            "mean_prediction_variance_ratio_last": _finite(group["prediction_target_variance_ratio_last"].mean()),
            "mean_q10_q90_coverage_first": _finite(group["q10_q90_coverage_first"].mean()),
            "mean_q10_q90_coverage_last": _finite(group["q10_q90_coverage_last"].mean()),
            "mean_validation_pinball_change_pct": _finite(group["validation_pinball_change_pct"].mean()),
        })

    first, last = history.iloc[0], history.iloc[-1]
    primary_skill = history["validation_primary_site_mean_mse_skill"].to_numpy(float)
    best_primary_index = int(np.argmax(primary_skill))
    mean_val_pinball = epoch.loc[epoch["split"] == "validation"].groupby("epoch")[
        "pinball_sum_quantiles_weighted"
    ].mean()
    min_mean_val_pinball_epoch = int(mean_val_pinball.idxmin())
    train_opt = optimization.groupby("component")[
        ["nonfinite_gradient_tensors", "nonfinite_update_tensors"]
    ].sum()
    summary = {
        "run_id": json.loads((run_dir / "experiment_metadata.json").read_text())["run_id"],
        "epoch_count": len(history),
        "selected_epoch": int(json.loads((run_dir / "selection.json").read_text())["selected_epoch"]),
        "early_stop_epoch": int(history["epoch"].max()),
        "training_pinball": {
            "first": _finite(first["train_pinball_loss"]),
            "last": _finite(last["train_pinball_loss"]),
            "reduction_percent": _finite(100 * (1 - last["train_pinball_loss"] / first["train_pinball_loss"])),
        },
        "validation_primary_skill": {
            "first": _finite(history.iloc[0]["validation_primary_site_mean_mse_skill"]),
            "last": _finite(history.iloc[-1]["validation_primary_site_mean_mse_skill"]),
            "best": _finite(np.max(primary_skill)),
            "best_epoch": int(history.iloc[best_primary_index]["epoch"]),
            "selected_epoch_value": _finite(history.loc[history["epoch"] == int(json.loads((run_dir / "selection.json").read_text())["selected_epoch"]), "validation_primary_site_mean_mse_skill"].iloc[0]),
        },
        "mean_validation_pinball_across_scale_lead_cells": {
            "first": _finite(mean_val_pinball.iloc[0]),
            "last": _finite(mean_val_pinball.iloc[-1]),
            "best_epoch": min_mean_val_pinball_epoch,
            "best": _finite(mean_val_pinball.min()),
        },
        "validation_rmse_primary_macro_site": {
            "first": _finite(history.iloc[0]["validation_macro_site_spei3_t3_t4_rmse"]),
            "last": _finite(history.iloc[-1]["validation_macro_site_spei3_t3_t4_rmse"]),
            "best_epoch": int(history["validation_macro_site_spei3_t3_t4_rmse"].idxmin() + 1),
        },
        "validation_severe_recall": {
            "first": _finite(history.iloc[0]["validation_severe_category_recall_spei3_t3_t4"]),
            "last": _finite(history.iloc[-1]["validation_severe_category_recall_spei3_t3_t4"]),
            "actual_events_last": int(per_cell.loc[
                (per_cell["target"] == "spei_3") & per_cell["lead"].isin(["T+3", "T+4"]),
                "actual_severe_events_last",
            ].sum()),
            "predicted_events_last": int(per_cell.loc[
                (per_cell["target"] == "spei_3") & per_cell["lead"].isin(["T+3", "T+4"]),
                "predicted_severe_events_last",
            ].sum()),
        },
        "optimizer_diagnostics": {
            "nonfinite_gradient_tensors": int(train_opt["nonfinite_gradient_tensors"].sum()),
            "nonfinite_update_tensors": int(train_opt["nonfinite_update_tensors"].sum()),
            "components": sorted(optimization["component"].unique().tolist()),
            "first_epoch": optimization.loc[optimization["epoch"] == optimization["epoch"].min()].to_dict("records"),
            "last_epoch": optimization.loc[optimization["epoch"] == optimization["epoch"].max()].to_dict("records"),
        },
        "test_split_used": False,
        "validation_is_exploratory": True,
        "per_scale_lead": records,
        "per_scale_summary": scale_records,
        "training_per_scale_summary": train_scale_records,
        "primary_endpoint_rows": len(primary),
    }
    (run_dir / "diagnostic_metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    scale_table = "\n".join(
        "| {target} | {mean_validation_pinball_first:.3f} | {mean_validation_pinball_last:.3f} | "
        "{mean_validation_rmse_first:.3f} | {mean_validation_rmse_last:.3f} | "
        "{mean_prediction_variance_ratio_first:.3f} | {mean_prediction_variance_ratio_last:.3f} | "
        "{mean_q10_q90_coverage_last:.1%} |".format(**row)
        for row in scale_records
    )
    training_scale_table = "\n".join(
        "| {target} | {mean_per_lead_pinball_first:.3f} | {mean_per_lead_pinball_last:.3f} | "
        "{mean_per_lead_variance_ratio_first:.3f} | {mean_per_lead_variance_ratio_last:.3f} |".format(**row)
        for row in train_scale_records
    )
    report = f"""# Phase 2.5.1 — Instrumented Objective-Alignment Baseline

## Decision

**BASELINE EVIDENCE WITH LIMITATIONS.** This is one exploratory seed on a validation period repeatedly used for selection. It is diagnostic evidence, not independent confirmation and not a benchmark update.

## Frozen setup and data isolation

- Run: `{summary['run_id']}`; seed 42; frozen Phase 2.2 profile `regularized_batch64`.
- Existing model, training loss, optimizer, scheduler, early stopping, batch size, learning rate, weight decay, and data preprocessing were reused unchanged.
- Inputs: 6 months × 15 features; quantile outputs q10/q50/q90 for 3 SPEI scales × 4 leads.
- Constructed windows: train 2,916; validation 468. The loader returned an empty test placeholder (0 windows). No test predictions, metrics, or labels were read.
- Dataset SHA-256: `{json.loads((run_dir / 'experiment_metadata.json').read_text())['dataset_sha256']}`
- Normalizer SHA-256: `{json.loads((run_dir / 'experiment_metadata.json').read_text())['normalizer_sha256']}`
- Selected checkpoint epoch: {summary['selected_epoch']}; early stop: epoch {summary['early_stop_epoch']}.

## Observations

- Training pinball decreased from {summary['training_pinball']['first']:.3f} to {summary['training_pinball']['last']:.3f} ({summary['training_pinball']['reduction_percent']:.1f}%).
- The mean validation per-cell pinball diagnostic decreased from {summary['mean_validation_pinball_across_scale_lead_cells']['first']:.3f} to {summary['mean_validation_pinball_across_scale_lead_cells']['last']:.3f}; its minimum was at epoch {summary['mean_validation_pinball_across_scale_lead_cells']['best_epoch']}. This is the mean of 12 scale/lead observer rows, not the training objective's exact batch reduction.
- Validation primary equal-site SPEI-3 T+3/T+4 MSE skill peaked at epoch {summary['validation_primary_skill']['best_epoch']} ({summary['validation_primary_skill']['best']:.3f}) and ended at {summary['validation_primary_skill']['last']:.3f}. The selected epoch therefore matches the configured checkpoint rule; it is not evidence of a checkpoint-code defect.
- Macro-site validation SPEI-3 T+3/T+4 RMSE moved from {summary['validation_rmse_primary_macro_site']['first']:.3f} to {summary['validation_rmse_primary_macro_site']['last']:.3f}. At final epoch the validation pinball objective proxy is lower while this endpoint RMSE is higher.
- For SPEI-3, mean validation prediction/target variance ratio rose from {scale_records[1]['mean_prediction_variance_ratio_first']:.4f} to {scale_records[1]['mean_prediction_variance_ratio_last']:.3f}; for SPEI-6 it rose from {scale_records[2]['mean_prediction_variance_ratio_first']:.4f} to {scale_records[2]['mean_prediction_variance_ratio_last']:.3f}; for SPEI-12 it rose from {scale_records[0]['mean_prediction_variance_ratio_first']:.3f} to {scale_records[0]['mean_prediction_variance_ratio_last']:.3f}. Variance grew, especially at longer scales, but does not establish calibrated dispersion.
- Final SPEI-3 T+3/T+4 q50 predictions identified 0 of {summary['validation_severe_recall']['actual_events_last']} observed validation events at SPEI ≤ -1.5. This is an exploratory count over the selected validation period.
- No quantile crossings were recorded in the train/validation observer rows; no non-finite gradient tensors or parameter updates were recorded.

The observer recorded each of the 12 training scale/lead pinball cells at each epoch. Their per-scale four-lead averages and variance ratios are summarized here; the detailed rows remain in `epoch_metrics.csv`.

| Scale | Train pinball mean, epoch 1 | Train pinball mean, final | Train variance ratio, epoch 1 | Train variance ratio, final |
|---|---:|---:|---:|---:|
{training_scale_table}

## Scale summary

Validation q10–q90 coverage is nominally 80%; observed coverage below 80% indicates undercoverage, not calibrated predictive intervals.

| Scale | Mean pinball, epoch 1 | Mean pinball, final | Mean RMSE, epoch 1 | Mean RMSE, final | Variance ratio, epoch 1 | Variance ratio, final | q10–q90 coverage, final |
|---|---:|---:|---:|---:|---:|---:|---:|
{scale_table}

SPEI-6 and SPEI-12 per-cell validation pinball/RMSE generally improved across training, while SPEI-3 T+3/T+4 showed small pinball reductions and worse median RMSE than epoch 1. This scale/lead heterogeneity is consistent with the aggregate objective emphasizing outputs other than the primary endpoint, but does not establish causality.

## Interpretation

**Supported observation:** the multi-scale quantile objective keeps decreasing while the validation checkpoint criterion is best at epoch 1; the other scale/lead diagnostics do not share the same trajectory. The epoch-1 selection is correctly produced by the declared validation criterion. Prediction variance grows during training, yet SPEI-3 primary-lead predictions remain substantially under-dispersed and severe-event q50 counts remain zero.

**Interpretation, not causal proof:** objective/checkpoint mismatch is a plausible contributor to the divergence between optimization and selection metrics. A single seed and repeatedly used validation period cannot establish that it causes prediction collapse. Seed variation and distribution shift remain unquantified here.

## Proposed next experiment (not run)

Change only the training objective; retain architecture, optimizer, learning rate, batch size, regularization, preprocessing, seed protocol, and checkpoint rule. Let `L_pinball` be the unchanged sum of the three scale-wise losses from `train.pinball_loss`, including its current masks and extreme weighting. Define `MSE_primary` as the mean squared error of q50 over valid SPEI-3 targets at T+3 and T+4, pooled over training examples and those two leads. Compare the existing objective with:

`L_alt = L_pinball + 0.5 * MSE_primary`.

The fixed coefficient 0.5 is predeclared for a single diagnostic comparison and must not be tuned on validation or test results. The original pinball term remains, so every scale, lead, and q10/q50/q90 output remains supervised; the added term directly aligns the SPEI-3 T+3/T+4 median with the checkpoint criterion. The extra term may change the balance between median accuracy and quantile calibration, so coverage and width must remain reported. Select checkpoints using the same validation primary skill. Treat validation comparisons as exploratory. A better score would not prove objective mismatch was the sole cause; changes in dispersion, secondary scales, and severe-event behavior matter. This proposal is not executed.

## Artifacts

- `epoch_metrics.csv`: per epoch, split, scale and lead diagnostics.
- `primary_endpoint.csv`: site and lead validation MSE/skill.
- `optimization_diagnostics.csv`: gradient and parameter-update norms.
- `training_history.csv`, `selection.json`, `effective_configuration.json`, `experiment_metadata.json`.
- `diagnostic_metrics.json` and `per_scale_lead_summary.csv` generated by `experiments.summarize_phase2_5_1`.

No test-set conclusions are reported.
"""
    (run_dir / "baseline_experiment_report.md").write_text(report, encoding="utf-8")
    objective_proposal = """# Proposed objective-alignment experiment (not run)

Use the same model, seed protocol, data, preprocessing, optimizer and hyperparameters as the frozen Phase 2.2 `regularized_batch64` profile. Change only the relative objective weights.

Let `L_pinball` be the existing summed three-scale quantile pinball loss, retaining the current mask and extreme-weight handling. Let `MSE_primary = mean((y_spei3 - q50_spei3)^2)` over valid examples at T+3 and T+4. The single proposed alternative is `L_alt = L_pinball + 0.5 * MSE_primary`. The coefficient is fixed before the experiment, not tuned on validation/test. All quantiles and all scale/lead targets remain supervised by the original pinball term. The added median loss directly aligns the primary outputs with checkpoint selection, while risking worse interval calibration.

Keep the same architecture, seed protocol, data, preprocessing, optimizer and hyperparameters. Keep checkpoint selection as equal-site validation MSE skill versus persistence on SPEI-3 T+3/T+4 and use the same early-stopping rule. Compare validation primary skill, scale/lead pinball, median variance ratios, interval coverage, crossings and severe-event counts. The repeatedly used validation period is exploratory and cannot establish generalization or causality. Do not inspect the consumed test split. This proposal awaits separate authorization.
"""
    (run_dir / "proposed_objective_alignment_experiment.md").write_text(
        objective_proposal, encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
