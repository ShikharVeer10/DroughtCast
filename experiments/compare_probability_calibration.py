"""Compare validation-only probability calibrators on the untouched test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from data.loader import load_site_table, validate_dataset_protocol
from models.forecaster import MultiScaleForecaster
from utils.artifacts import validate_checkpoint_metadata
from utils.probability import (
    brier_score,
    calibration_slope_intercept,
    expected_calibration_error,
    quantile_risk_below_threshold,
)

SCALES = ("spei_3", "spei_6", "spei_12")


def fit_models(x: np.ndarray, y: np.ndarray) -> dict:
    result = {"raw": lambda values: np.clip(values, 0, 1)}
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(x, y)
    result["isotonic"] = lambda values: iso.predict(values)
    logistic = LogisticRegression(solver="lbfgs").fit(
        np.clip(x, 1e-6, 1 - 1e-6).reshape(-1, 1), y
    )
    result["logistic"] = lambda values: logistic.predict_proba(
        np.clip(values, 1e-6, 1 - 1e-6).reshape(-1, 1)
    )[:, 1]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="results/ablations/quantile_cqr/config.yaml"
    )
    parser.add_argument(
        "--checkpoint", default="results/ablations/quantile_cqr/model_seed_42.pt"
    )
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument(
        "--output",
        default="results/probability_calibration/calibration_comparison_seed_42.csv",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    protocol = validate_dataset_protocol()
    data = load_site_table(
        cfg["data"]["path"],
        cfg["data"]["features"],
        cfg["data"]["seq_len"],
        cfg["data"]["forecast_steps"],
        cfg["data"]["metadata_path"],
        [],
        cfg["data"].get("site_filter"),
        cfg["model"].get("input_extras"),
        cfg["model"].get("static_context", False),
        False,
        cfg.get("sites"),
    )
    cfg["data"]["input_dim"] = data["test"].X.shape[-1]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(checkpoint, cfg)
    model = MultiScaleForecaster(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        val_output, _ = model(data["val"].X)
        test_output, _ = model(data["test"].X)
    rows = []
    for index, scale in enumerate(SCALES):
        val_q, test_q = val_output[scale].numpy(), test_output[scale].numpy()
        val_y, test_y = (
            data["val"].Y.numpy()[:, index],
            data["test"].Y.numpy()[:, index],
        )
        val_mask, test_mask = (
            data["val"].mask.numpy()[:, index],
            data["test"].mask.numpy()[:, index],
        )
        for lead in range(test_q.shape[1]):
            valid_val = val_mask[:, lead] & np.isfinite(val_q[:, lead]).all(axis=1)
            valid_test = test_mask[:, lead] & np.isfinite(test_q[:, lead]).all(axis=1)
            raw_val = quantile_risk_below_threshold(
                val_q[valid_val, lead], args.threshold
            )
            raw_test = quantile_risk_below_threshold(
                test_q[valid_test, lead], args.threshold
            )
            val_event = (val_y[valid_val, lead] <= args.threshold).astype(int)
            test_event = (test_y[valid_test, lead] <= args.threshold).astype(int)
            for method, predictor in fit_models(raw_val, val_event).items():
                probability = np.asarray(predictor(raw_test), dtype=float)
                rows.append(
                    {
                        "scale": scale,
                        "lead": f"T+{lead + 1}",
                        "method": method,
                        "n_validation": len(val_event),
                        "n_test": len(test_event),
                        "brier": brier_score(probability, test_event),
                        "ece": expected_calibration_error(probability, test_event),
                        "calibration_intercept": calibration_slope_intercept(
                            probability, test_event
                        )["intercept"],
                        "calibration_slope": calibration_slope_intercept(
                            probability, test_event
                        )["slope"],
                    }
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(
        f"\nSaved comparison to {output}; protocol schema {protocol['schema_version']}"
    )


if __name__ == "__main__":
    main()
