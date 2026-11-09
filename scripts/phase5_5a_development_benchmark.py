"""Run the Phase 5.5A development-only benchmark (never loads final evaluation)."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loader import load_site_table
from data.protocol import MODEL_FEATURES, PHASE5_1_1_SPLITS, SPEI_CALIBRATION_END
from evaluation.baselines import fit_classical, persistence, seasonal_predictions
from models.forecaster import MultiScaleForecaster
from train import pinball_loss

OUT = ROOT / "results" / "maharashtra" / "phase5" / "phase5_5a_development_benchmark"
DATA = ROOT / "data" / "maharashtra" / "monthly_sites.csv"
SCALES = ("spei_3", "spei_6", "spei_12")
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> list[dict]:
    rows = []
    for scale, scale_name in enumerate(SCALES):
        for lead in range(target.shape[2]):
            valid = mask[:, scale, lead] & np.isfinite(prediction[:, scale, lead])
            y, p = target[valid, scale, lead], prediction[valid, scale, lead]
            row = {"scale": scale_name, "lead": lead + 1, "n": len(y),
                   "sites": 0, "origins": 0, "rmse": None, "mae": None,
                   "mean_bias": None, "pearson": None}
            if len(y):
                row.update(rmse=float(np.sqrt(np.mean((p - y) ** 2))),
                           mae=float(np.mean(np.abs(p - y))),
                           mean_bias=float(np.mean(p - y)))
                row["pearson"] = float(np.corrcoef(p, y)[0, 1]) if len(y) > 1 and np.std(p) and np.std(y) else None
            rows.append(row)
    return rows


def prediction_rows(name, prediction, dataset):
    rows = []
    target = dataset.Y.numpy()
    mask = dataset.mask.numpy()
    for i, meta in enumerate(dataset.metadata):
        for scale, scale_name in enumerate(SCALES):
            for lead in range(target.shape[2]):
                if not mask[i, scale, lead] or not np.isfinite(prediction[i, scale, lead]):
                    continue
                rows.append({
                    "model": name, "site": meta["site"], "forecast_origin": meta["dates"][-1],
                    "target_month": meta["target_dates"][lead], "scale": scale_name,
                    "lead": lead + 1, "observed": float(target[i, scale, lead]),
                    "prediction": float(prediction[i, scale, lead]),
                })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    features = list(MODEL_FEATURES)
    data = load_site_table(
        DATA, features=features, seq_len=6, forecast_steps=4, metadata_path=None,
        site_filter=None, split_config=PHASE5_1_1_SPLITS, include_splits=("train", "val")
    )
    train_set, val_set = data["train"], data["val"]
    predictions: dict[str, np.ndarray] = {
        "persistence": persistence(val_set.metadata, 4),
        "seasonal_climatology": seasonal_predictions(train_set, val_set),
        "ridge": fit_classical(train_set, val_set, "ridge"),
        "xgboost": fit_classical(train_set, val_set, "xgboost"),
    }
    config = {
        "data": {"seq_len": 6, "forecast_steps": 4, "features": features, "input_dim": len(features)},
        "model": {"encoder": "lstm", "attention": "shared", "heads": "direct",
                  "input_dim": len(features), "hidden_dim_1": 64, "hidden_dim_2": 32,
                  "dropout": 0.2, "quantiles": [0.1, 0.5, 0.9]},
        "training": {"batch_size": 32, "learning_rate": 0.001, "weight_decay": 0.0001,
                     "task_weighting": "fixed"},
    }
    model = MultiScaleForecaster(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    best_state, best_loss = None, float("inf")
    for _ in range(8):
        model.train()
        optimizer.zero_grad()
        output, _ = model(train_set.X)
        stacked = torch.stack([output[scale] for scale in SCALES], dim=1)
        loss = pinball_loss(stacked, train_set.Y, train_set.mask, [0.1, 0.5, 0.9])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_output, _ = model(val_set.X)
            val_stack = torch.stack([val_output[scale] for scale in SCALES], dim=1)
            val_loss = float(pinball_loss(val_stack, val_set.Y, val_set.mask, [0.1, 0.5, 0.9]))
        if val_loss < best_loss:
            best_loss, best_state = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        neural_output, _ = model(val_set.X)
    predictions["shared_lstm"] = np.stack([neural_output[scale][..., 1].numpy() for scale in SCALES], axis=1)

    all_metrics, all_predictions = [], []
    target, mask = val_set.Y.numpy(), val_set.mask.numpy()
    for name, prediction in predictions.items():
        rows = metrics(prediction, target, mask)
        for row in rows:
            valid_meta = [val_set.metadata[i] for i in range(len(val_set)) if mask[i, SCALES.index(row["scale"]), row["lead"] - 1]]
            row.update(model=name, sites=len({m["site"] for m in valid_meta}),
                       origins=len({m["dates"][-1] for m in valid_meta}))
        all_metrics.extend(rows)
        all_predictions.extend(prediction_rows(name, prediction, val_set))
    metric_frame = pd.DataFrame(all_metrics)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(OUT / "development_predictions.csv", index=False)
    baseline = metric_frame[metric_frame["model"].isin(["persistence", "seasonal_climatology"])][
        ["scale", "lead", "model", "rmse", "mae", "mean_bias", "pearson"]
    ].rename(columns={"model": "baseline"})
    comparisons = metric_frame[~metric_frame["model"].isin(["persistence", "seasonal_climatology"])].merge(
        baseline, on=["scale", "lead"], suffixes=("_model", "_baseline")
    )
    comparisons["rmse_difference"] = comparisons["rmse_model"] - comparisons["rmse_baseline"]
    comparisons.to_csv(OUT / "baseline_comparisons.csv", index=False)
    best_model = metric_frame.groupby("model", as_index=False)["rmse"].mean().sort_values("rmse").iloc[0]["model"]
    (OUT / "experiment_configuration.json").write_text(json.dumps({
        "seed": SEED, "splits": PHASE5_1_1_SPLITS, "spei_calibration_end": SPEI_CALIBRATION_END,
        "embargo_months": 22, "test_loaded": False, "test_used": False,
        "train_samples": len(train_set), "validation_samples": len(val_set),
        "models": list(predictions), "calibration_intervals": False,
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "model_inventory.json").write_text(json.dumps({
        "persistence": {}, "seasonal_climatology": {"fit": "training only"},
        "ridge": {"alpha": 10.0}, "xgboost": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "random_state": 42},
        "shared_lstm": config | {"epochs": 8, "selection": "validation pinball loss"},
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "matched_sample_report.md").write_text(
        f"# Matched sample report\n\nAll models used the same **{len(train_set)}** training and **{len(val_set)}** validation windows. "
        "Metrics are reported separately by scale and lead; no final-evaluation rows were loaded.\n", encoding="utf-8")
    (OUT / "model_selection_rationale.md").write_text(
        f"# Development selection\n\nSelected candidate: **{best_model}** by lowest mean validation RMSE across scale/lead cells. "
        "This is development-only evidence and is not independent confirmation.\n", encoding="utf-8")
    selected = OUT / "selected_model_config.json"
    selected.write_text(json.dumps({"selected_model": best_model, "seed": SEED, "validation_only": True}, indent=2) + "\n", encoding="utf-8")
    (OUT / "frozen_candidate_manifest.json").write_text(json.dumps({
        "selected_model": best_model, "architecture_or_parameters_frozen": True,
        "prospective_outcome_change_rule": "No changes after prospective outcomes are observed.",
        "test_period_used": False,
        "artifacts": {"selected_model_config.json": sha256(selected)},
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "limitations_report.md").write_text(
        "# Limitations\n\nDevelopment results only; validation has influenced selection. The 2017–2025 period was not loaded "
        "and remains contaminated from prior work, so no independent test or superiority claim is made. No calibrated "
        "intervals or probabilities are reported. Overlapping windows are dependent and require block-aware inference.\n",
        encoding="utf-8")
    (OUT / "verification_report.md").write_text(
        f"# Verification report\n\nModels run: {', '.join(predictions)}. Train windows: {len(train_set)}. "
        f"Validation windows: {len(val_set)}. Final evaluation loaded: **no**.\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.name != "sha256_manifest.json")
    (OUT / "sha256_manifest.json").write_text(json.dumps({path.name: sha256(path) for path in files}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
