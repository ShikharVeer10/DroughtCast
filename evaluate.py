"""Evaluate model and classical baselines once on the untouched test partition."""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

from data.loader import load_site_table
from evaluation.baselines import fit_classical, persistence, seasonal_predictions
from models.forecaster import MultiScaleForecaster
from utils.artifacts import validate_checkpoint_metadata
from utils.metrics import (calculate_rmse, calculate_mae, calculate_nse,calculate_kge, drought_event_metrics, skill_score_persistence)
from utils.stats import (count_drought_episodes, dm_hac, effective_sample_size,
                         holm_adjust, paired_block_bootstrap_rmse_ci,
                         wilcoxon_signed_rank_test)

SCALES = ("spei_3", "spei_6", "spei_12")


def summarize_predictions(predictions, targets, mask, metadata):
    results = {}
    for scale, name in enumerate(SCALES):
        results[name] = {}
        for lead in range(targets.shape[-1]):
            valid = mask[:, scale, lead] & np.isfinite(predictions[:, scale, lead])
            y, p = targets[valid, scale, lead], predictions[valid, scale, lead]
            if not len(y):
                continue
            results[name][f"T+{lead+1}"] = {
                "n": int(len(y)),
                "rmse": calculate_rmse(p, y),
                "mae": calculate_mae(p, y),
                "nse": calculate_nse(p, y),
                "kge_2012": calculate_kge(p, y),
                "drought_event_moderate": drought_event_metrics(p, y, threshold=-1.0),
                "drought_event_severe": drought_event_metrics(p, y, threshold=-1.5),
                "drought_event_extreme": drought_event_metrics(p, y, threshold=-2.0),
            }
        per_site = {}
        site_ids = np.asarray([m["site"] for m in metadata])
        for site in sorted(set(site_ids)):
            rows = site_ids == site
            per_site[site] = {}
            for lead in range(targets.shape[-1]):
                valid = rows & mask[:, scale, lead] & np.isfinite(predictions[:, scale, lead])
                y, p = targets[valid, scale, lead], predictions[valid, scale, lead]
                if len(y):
                    per_site[site][f"T+{lead+1}"] = {
                        "n": int(len(y)),
                        "rmse": calculate_rmse(p, y),
                        "mae": calculate_mae(p, y),
                        "nse": calculate_nse(p, y),
                        "kge_2012": calculate_kge(p, y)
                    }
        results[name]["by_site"] = per_site
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/current_model.pt")
    parser.add_argument("--output", default="results/test_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--reuse-baseline-report", help="Reuse deterministic summaries from a completed evaluation JSON")
    parser.add_argument("--holdout-region", action="append", default=[])
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data = load_site_table(cfg["data"]["path"], cfg["data"].get("features"),
        cfg["data"]["seq_len"], cfg["data"]["forecast_steps"], cfg["data"].get("metadata_path", "data/meta.json"),
        args.holdout_region, cfg["data"].get("site_filter"),
        cfg["model"].get("input_extras"), cfg["model"].get("static_context", False),
        cfg["model"].get("heads", cfg["model"].get("head_mode", "direct")) == "residual",
        cfg.get("sites"))
    train, test = data["train"], data["test"]
    if not len(test):
        raise RuntimeError("No valid test windows; data must cover 2017 onward")
    cfg["data"]["input_dim"] = test.X.shape[-1]
    model = MultiScaleForecaster(cfg)
    checkpoint_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(checkpoint_state, cfg)
    model.load_state_dict(checkpoint_state.get("model", checkpoint_state))
    model.eval()
    with torch.no_grad():
        model_prediction, _ = model(test.X,
                                    last_spei=test.last_spei,
                                    static=test.static)
    quantiles = cfg["model"].get("quantiles")
    calibration_report = None
    if quantiles:
        from utils.conformal import calibrate_cqr, apply_cqr, interval_metrics, crps_from_quantiles
        val = data["val"]
        with torch.no_grad():
            val_prediction = model(val.X, last_spei=val.last_spei, static=val.static)[0]
        calibrated = {}
        for scale_idx, scale_name in enumerate(SCALES):
            calibrated[scale_name] = calibrate_cqr(
                val_prediction[scale_name].numpy(), val.Y[:, scale_idx].numpy(),
                val.mask[:, scale_idx].numpy(), quantiles)
        calibration_report = {"method": "split CQR; validation only", "scales": {}}
        for scale_idx, scale_name in enumerate(SCALES):
            pred = model_prediction[scale_name].numpy()
            lower, upper = apply_cqr(pred, calibrated[scale_name])
            y = test.Y[:, scale_idx].numpy()
            mask_scale = test.mask[:, scale_idx].numpy()
            val_pred = val_prediction[scale_name].numpy()
            val_y = val.Y[:, scale_idx].numpy()
            val_mask = val.mask[:, scale_idx].numpy()
            val_lower, val_upper = apply_cqr(val_pred, calibrated[scale_name])
            metrics = {}
            for lead in range(y.shape[1]):
                valid = mask_scale[:, lead]
                val_valid = val_mask[:, lead]
                before = interval_metrics(pred[valid, lead, 0], pred[valid, lead, 2], y[valid, lead])
                after = interval_metrics(lower[valid, lead], upper[valid, lead], y[valid, lead])
                metrics[f"T+{lead+1}"] = {
                    "validation_before": interval_metrics(
                        val_pred[val_valid, lead, 0], val_pred[val_valid, lead, 2], val_y[val_valid, lead]),
                    "validation_after": interval_metrics(
                        val_lower[val_valid, lead], val_upper[val_valid, lead], val_y[val_valid, lead]),
                    "test_before": before,
                    "test_after": after,
                    "calibration_quantile": calibrated[scale_name][lead],
                    "crps_approx": crps_from_quantiles(pred[valid, lead:lead+1, :], y[valid, lead:lead+1], quantiles)
                }
            calibration_report["scales"][scale_name] = metrics
    targets, mask = test.Y.numpy(), test.mask.numpy()
    if quantiles:
        model_array = np.stack([model_prediction[k][:, :, 1].numpy() for k in SCALES], axis=1)
    else:
        model_array = np.stack([model_prediction[k].numpy() for k in SCALES], axis=1)

    pers_preds = persistence(test.metadata, cfg["data"]["forecast_steps"])
    climatology_preds = seasonal_predictions(train, test)

    baseline_predictions = {
        "persistence": pers_preds,
        "climatology": climatology_preds,
    }
    cached_baseline_results = None
    if args.reuse_baseline_report:
        cached = json.loads(Path(args.reuse_baseline_report).read_text(encoding="utf-8"))
        cached_baseline_results = {k: cached["methods"][k] for k in ("ridge", "hist_gradient_boosting", "xgboost", "var", "gru", "tcn") if k in cached.get("methods", {})}
    if not args.skip_classical and cached_baseline_results is None:
        for name in ("ridge", "hist_gradient_boosting", "xgboost", "var", "gru", "tcn"):
            try:
                baseline_predictions[name] = fit_classical(train, test, name)
            except Exception as exc:
                warnings.warn(f"Skipping baseline {name!r}: {exc}", stacklevel=2)

    all_predictions = {"droughtcast": model_array, **baseline_predictions}
    for method, prediction in all_predictions.items():
        for other, other_prediction in all_predictions.items():
            if method < other and np.array_equal(prediction, other_prediction, equal_nan=True):
                raise RuntimeError(f"Methods {method!r} and {other!r} produced identical predictions")
    results = {name: summarize_predictions(pred, targets, mask, test.metadata)
               for name, pred in all_predictions.items()}
    if cached_baseline_results is not None:
        results.update(cached_baseline_results)

    # Standard Hydrological Skill Score over Persistence (SS_pers = 1 - MSE_model / MSE_pers).
    # Cached classical summaries already contain their skill fields, so only
    # methods with predictions in this process are recomputed.
    for method, values in results.items():
        if method == "persistence":
            continue
        if method not in all_predictions:
            continue
        for scale_idx, scale in enumerate(SCALES):
            for lead_idx, lead in enumerate(f"T+{i}" for i in range(1, cfg["data"]["forecast_steps"] + 1)):
                if lead in values[scale] and lead in results["persistence"][scale]:
                    valid = mask[:, scale_idx, lead_idx]
                    y_true = targets[valid, scale_idx, lead_idx]
                    y_pred = all_predictions[method][valid, scale_idx, lead_idx]
                    y_pers = pers_preds[valid, scale_idx, lead_idx]
                    values[scale][lead]["skill_vs_persistence"] = skill_score_persistence(y_pred, y_true, y_pers)

    # Statistical significance testing: compare every available method with
    # persistence and seasonal climatology using monthly site-panel losses.
    inference = {}
    for scale, scale_name in enumerate(SCALES):
        inference[scale_name] = {}
        for lead in range(targets.shape[-1]):
            valid = mask[:, scale, lead]
            y = targets[valid, scale, lead]
            sample_meta = [m for i, m in enumerate(test.metadata) if valid[i]]
            group_ids = [f"{m['site']}:{m['target_dates'][lead][:4]}" for m in sample_meta]
            month_keys = sorted({m["target_dates"][lead][:7] for m in sample_meta})

            def panel_losses(prediction):
                rows = {month: [] for month in month_keys}
                for observed, predicted, meta in zip(y, prediction[valid, scale, lead], sample_meta):
                    rows[meta["target_dates"][lead][:7]].append((predicted - observed) ** 2)
                return np.asarray([np.mean(rows[month]) for month in month_keys], dtype=float)

            droughtcast_losses = panel_losses(model_array)
            persistence_losses = panel_losses(pers_preds)
            climatology_losses = panel_losses(climatology_preds)
            inference[scale_name][f"T+{lead+1}"] = {
                "wilcoxon_vs_persistence": wilcoxon_signed_rank_test(
                    (model_array[valid, scale, lead] - y) ** 2,
                    (pers_preds[valid, scale, lead] - y) ** 2),
                "wilcoxon_vs_climatology": wilcoxon_signed_rank_test(
                    (model_array[valid, scale, lead] - y) ** 2,
                    (climatology_preds[valid, scale, lead] - y) ** 2),
                "primary_site_year_block_bootstrap_vs_persistence": paired_block_bootstrap_rmse_ci(
                    model_array[valid, scale, lead], pers_preds[valid, scale, lead],
                    y, group_ids),
                "secondary_pooled_wilcoxon_vs_persistence": wilcoxon_signed_rank_test(
                    (model_array[valid, scale, lead] - y) ** 2,
                    (pers_preds[valid, scale, lead] - y) ** 2),
                "dm_vs_persistence_squared_error_monthly_panel": dm_hac(
                    droughtcast_losses, persistence_losses, horizon=lead + 1,
                    accumulation_scale=(3, 6, 12)[scale]),
                "paired_block_bootstrap_vs_persistence": paired_block_bootstrap_rmse_ci(
                    model_array[valid, scale, lead], pers_preds[valid, scale, lead],
                    y, group_ids),
                "test_effective_sample_size": effective_sample_size(droughtcast_losses),
                "independent_drought_episodes": count_drought_episodes(
                    [m for i, m in enumerate(test.metadata) if valid[i]],
                    targets[valid], mask[valid], scale),
                "dm_vs_climatology_squared_error_monthly_panel": dm_hac(
                    droughtcast_losses, climatology_losses, horizon=lead + 1,
                    accumulation_scale=(3, 6, 12)[scale]),
            }
            for method, prediction in all_predictions.items():
                if method in {"droughtcast", "persistence", "seasonal_climatology", "climatology"}:
                    continue
                method_losses = panel_losses(prediction)
                inference[scale_name][f"T+{lead+1}"][f"dm_{method}_vs_persistence"] = dm_hac(
                    method_losses, persistence_losses, horizon=lead + 1,
                    accumulation_scale=(3, 6, 12)[scale])

    dm_entries = []
    for scale_values in inference.values():
        for lead_values in scale_values.values():
            for name, result in lead_values.items():
                if name.startswith("dm_") and np.isfinite(result.get("p_value", np.nan)):
                    dm_entries.append((result, result["p_value"]))
    for adjusted, (result, _) in zip(
        holm_adjust([p for _, p in dm_entries]), dm_entries
    ):
        result["p_value_holm"] = adjusted

    report = {"seed": args.seed, "checkpoint": args.checkpoint, "holdout_regions": args.holdout_region,
              "test_samples": len(test),
              "unique_target_months": len({
                  m["target_dates"][0][:7] for m in test.metadata
              }),
              "methods": results, "paired_inference": inference}
    if calibration_report is not None:
        report["probabilistic"] = calibration_report
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(f"Saved evaluation metrics to {output}")


if __name__ == "__main__":
    main()
