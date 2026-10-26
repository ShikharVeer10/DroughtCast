"""Fit validation-only event calibration and evaluate on the test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from data.loader import load_site_table
from models.forecaster import MultiScaleForecaster
from utils.artifacts import validate_checkpoint_metadata
from utils.probability import (brier_score, calibration_slope_intercept,
                               expected_calibration_error, fit_isotonic_calibrator,
                               reliability_table)

SCALES = ("spei_3", "spei_6", "spei_12")
QUANTILES = [0.1, 0.5, 0.9]


def risk_below_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    q10, q50, q90 = [values[..., i] for i in range(3)]
    result = np.empty_like(q50, dtype=float)
    result[threshold <= q10] = .1
    result[threshold >= q90] = .9
    low = (threshold > q10) & (threshold <= q50)
    high = (threshold > q50) & (threshold < q90)
    result[low] = .1 + .4 * (threshold - q10[low]) / (q50[low] - q10[low] + 1e-8)
    result[high] = .5 + .4 * (threshold - q50[high]) / (q90[high] - q50[high] + 1e-8)
    return np.clip(result, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="results/ablations/quantile_cqr/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-output")
    parser.add_argument("--threshold", type=float, default=-1.0)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data = load_site_table(cfg["data"]["path"], cfg["data"].get("features"),
        cfg["data"]["seq_len"], cfg["data"]["forecast_steps"],
        cfg["data"].get("metadata_path"), [], cfg["data"].get("site_filter"),
        cfg["model"].get("input_extras"), cfg["model"].get("static_context", False),
        False, cfg.get("sites"))
    cfg["data"]["input_dim"] = data["test"].X.shape[-1]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(state, cfg)
    model = MultiScaleForecaster(cfg)
    model.load_state_dict(state["model"])
    model.eval()
    with torch.no_grad():
        val_output, _ = model(data["val"].X)
        test_output, _ = model(data["test"].X)
    report = {"checkpoint": args.checkpoint, "threshold": args.threshold, "method": "validation-only isotonic calibration", "scales": {}}
    prediction_rows = []
    for index, scale in enumerate(SCALES):
        val_q = val_output[scale].numpy()
        test_q = test_output[scale].numpy()
        val_y = data["val"].Y.numpy()[:, index]
        test_y = data["test"].Y.numpy()[:, index]
        val_mask = data["val"].mask.numpy()[:, index]
        test_mask = data["test"].mask.numpy()[:, index]
        scale_report = {}
        for lead in range(test_q.shape[1]):
            valid_val = val_mask[:, lead] & np.isfinite(val_q[:, lead]).all()
            valid_test = test_mask[:, lead] & np.isfinite(test_q[:, lead]).all()
            raw_val = risk_below_threshold(val_q[valid_val, lead], args.threshold)
            raw_test = risk_below_threshold(test_q[valid_test, lead], args.threshold)
            val_event = (val_y[valid_val, lead] <= args.threshold).astype(int)
            test_event = (test_y[valid_test, lead] <= args.threshold).astype(int)
            calibrator = fit_isotonic_calibrator(raw_val, val_event)
            calibrated = calibrator.predict(raw_test)
            for row_index, sample_index in enumerate(np.flatnonzero(valid_test)):
                prediction_rows.append({
                    "site": data["test"].metadata[sample_index]["site"],
                    "date": data["test"].metadata[sample_index]["target_dates"][lead],
                    "lead": lead + 1, "scale": scale,
                    "raw_probability": float(raw_test[row_index]),
                    "calibrated_probability": float(calibrated[row_index]),
                    "observed_event": int(test_event[row_index]),
                })
            scale_report[f"T+{lead + 1}"] = {
                "n_validation": int(len(val_event)), "n_test": int(len(test_event)),
                "raw_brier": brier_score(raw_test, test_event),
                "calibrated_brier": brier_score(calibrated, test_event),
                "raw_ece": expected_calibration_error(raw_test, test_event),
                "calibrated_ece": expected_calibration_error(calibrated, test_event),
                "raw_calibration": calibration_slope_intercept(raw_test, test_event),
                "calibrated_calibration": calibration_slope_intercept(calibrated, test_event),
                "reliability": reliability_table(calibrated, test_event),
            }
        report["scales"][scale] = scale_report
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.predictions_output:
        import csv
        predictions_output = Path(args.predictions_output)
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with predictions_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=prediction_rows[0].keys())
            writer.writeheader()
            writer.writerows(prediction_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
