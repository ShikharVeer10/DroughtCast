"""Evaluate model and classical baselines once on the untouched test partition."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from data.loader import load_site_table
from evaluation.baselines import fit_classical, persistence, seasonal_predictions
from models.forecaster import MultiScaleForecaster
from utils.metrics import (calculate_rmse, calculate_mae, calculate_nse,
                           calculate_kge, drought_event_metrics)
from utils.stats import dm_hac, paired_block_bootstrap_rmse_ci

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
                "n": int(len(y)), "rmse": calculate_rmse(p, y), "mae": calculate_mae(p, y),
                "nse": calculate_nse(p, y), "kge_2012": calculate_kge(p, y),
                "drought_event": drought_event_metrics(p, y),
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
                    per_site[site][f"T+{lead+1}"] = {"n": int(len(y)),
                        "rmse": calculate_rmse(p, y), "mae": calculate_mae(p, y),
                        "nse": calculate_nse(p, y), "kge_2012": calculate_kge(p, y)}
        results[name]["by_site"] = per_site
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_research_model.pt")
    parser.add_argument("--output", default="results/test_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--reuse-baseline-report", help="Reuse deterministic Ridge/HGB summaries from a completed evaluation JSON")
    parser.add_argument("--holdout-region", action="append", default=[])
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data = load_site_table(cfg["data"]["path"], cfg["data"].get("features"),
        cfg["data"]["seq_len"], cfg["data"]["forecast_steps"], cfg["data"].get("metadata_path", "data/meta.json"),
        args.holdout_region, cfg["data"].get("site_filter"))
    train, test = data["train"], data["test"]
    if not len(test):
        raise RuntimeError("No valid test windows; data must cover 2017 onward")
    cfg["data"]["input_dim"] = test.X.shape[-1]
    model = MultiScaleForecaster(cfg)
    checkpoint_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_state.get("model", checkpoint_state))
    model.eval()
    with torch.no_grad():
        model_prediction, _ = model(test.X)
    quantiles = cfg["model"].get("quantiles")
    calibration_report = None
    if quantiles:
        from utils.conformal import calibrate_cqr, apply_cqr, interval_metrics, crps_from_quantiles
        val = data["val"]
        with torch.no_grad():
            val_prediction = model(val.X)[0]
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
            metrics = {}
            for lead in range(y.shape[1]):
                valid = mask_scale[:, lead]
                before = interval_metrics(pred[valid, lead, 0], pred[valid, lead, 2], y[valid, lead])
                after = interval_metrics(lower[valid, lead], upper[valid, lead], y[valid, lead])
                metrics[f"T+{lead+1}"] = {"before": before, "after": after,
                "calibration_quantile": calibrated[scale_name][lead],
                    "crps_approx": crps_from_quantiles(pred[valid, lead:lead+1, :],
                                                       y[valid, lead:lead+1], quantiles)}
            calibration_report["scales"][scale_name] = metrics
    targets, mask = test.Y.numpy(), test.mask.numpy()
    if quantiles:
        model_array = np.stack([model_prediction[k][:, :, 1].numpy() for k in SCALES], axis=1)
    else:
        model_array = np.stack([model_prediction[k].numpy() for k in SCALES], axis=1)
    baseline_predictions = {
        "persistence": persistence(test.metadata, cfg["data"]["forecast_steps"]),
        "seasonal_climatology": seasonal_predictions(train, test),
    }
    cached_baseline_results = None
    if args.reuse_baseline_report:
        cached = json.loads(Path(args.reuse_baseline_report).read_text(encoding="utf-8"))
        cached_baseline_results = {k: cached["methods"][k] for k in ("ridge", "hist_gradient_boosting")}
    if not args.skip_classical and cached_baseline_results is None:
        baseline_predictions["ridge"] = fit_classical(train, test, "ridge")
        baseline_predictions["hist_gradient_boosting"] = fit_classical(train, test, "hist_gradient_boosting")
    all_predictions = {"droughtcast": model_array, **baseline_predictions}
    results = {name: summarize_predictions(pred, targets, mask, test.metadata)
               for name, pred in all_predictions.items()}
    if cached_baseline_results is not None:
        results.update(cached_baseline_results)
    # Add scale/lead skill against persistence to each method.
    persistence_results = results["persistence"]
    for method, values in results.items():
        if method == "persistence":
            continue
        for scale in SCALES:
            for lead in (f"T+{i}" for i in range(1, cfg["data"]["forecast_steps"] + 1)):
                if lead in values[scale] and lead in persistence_results[scale]:
                    values[scale][lead]["skill_vs_persistence"] = 1 - values[scale][lead]["rmse"] / persistence_results[scale][lead]["rmse"]
    inference = {}
    for scale, scale_name in enumerate(SCALES):
        inference[scale_name] = {}
        for lead in range(targets.shape[-1]):
            valid = mask[:, scale, lead]
            y = targets[valid, scale, lead]
            p = model_array[valid, scale, lead]
            b = baseline_predictions["persistence"][valid, scale, lead]
            sample_meta = [m for i, m in enumerate(test.metadata) if valid[i]]
            group_ids = [f"{m['site']}:{m['target_dates'][lead][:4]}" for m in sample_meta]
            # Aggregate each month's loss across sites first; HAC then models monthly
            # serial dependence rather than treating pooled site windows as one series.
            monthly_rows = {}
            for observed, predicted, base, meta in zip(y, p, b, sample_meta):
                month = meta["target_dates"][lead][:7]
                monthly_rows.setdefault(month, [[], []])
                monthly_rows[month][0].append((predicted - observed) ** 2)
                monthly_rows[month][1].append((base - observed) ** 2)
            month_keys = sorted(monthly_rows)
            model_month_losses = [np.mean(monthly_rows[k][0]) for k in month_keys]
            base_month_losses = [np.mean(monthly_rows[k][1]) for k in month_keys]
            inference[scale_name][f"T+{lead+1}"] = {
                "dm_vs_persistence_squared_error_monthly_panel": dm_hac(model_month_losses, base_month_losses, horizon=lead+1),
                "paired_block_bootstrap_vs_persistence": paired_block_bootstrap_rmse_ci(p, b, y, group_ids),
            }
    report = {"seed": args.seed, "checkpoint": args.checkpoint, "holdout_regions": args.holdout_region,
              "test_samples": len(test), "methods": results, "paired_inference": inference}
    if calibration_report is not None:
        report["probabilistic"] = calibration_report
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    compact = {}
    for method, scales in results.items():
        compact[method] = {scale: {lead: {k: metrics.get(k) for k in ("n", "rmse", "mae", "nse", "kge_2012", "skill_vs_persistence")}
                                  for lead, metrics in vals.items() if lead.startswith("T+")}
                           for scale, vals in scales.items()}
    print(json.dumps({"seed": args.seed, "test_samples": len(test), "summary": compact}, indent=2, allow_nan=True))
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
