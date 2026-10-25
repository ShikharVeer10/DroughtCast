"""Generate site-level probabilistic drought maps from a quantile checkpoint.

The lower-tail risk is interpolated from the trained 0.1/0.5/0.9 quantiles.
The conformal component calibrates interval coverage, not the event
probability itself; the figure labels this distinction explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from data.loader import load_site_table
from models.forecaster import MultiScaleForecaster
from utils.artifacts import validate_checkpoint_metadata
from utils.conformal import calibrate_cqr, apply_cqr, interval_metrics

SCALES = ("spei_3", "spei_6", "spei_12")
QUANTILES = [0.1, 0.5, 0.9]


def risk_below_threshold(quantile_values: np.ndarray, threshold: float) -> np.ndarray:
    """Piecewise-linear CDF estimate from the three trained quantiles."""
    q10, q50, q90 = [quantile_values[..., i] for i in range(3)]
    out = np.empty_like(q50, dtype=float)
    out[threshold <= q10] = 0.1
    out[threshold >= q90] = 0.9
    middle_low = (threshold > q10) & (threshold <= q50)
    middle_high = (threshold > q50) & (threshold < q90)
    out[middle_low] = 0.1 + 0.4 * (threshold - q10[middle_low]) / (
        q50[middle_low] - q10[middle_low] + 1e-8)
    out[middle_high] = 0.5 + 0.4 * (threshold - q50[middle_high]) / (
        q90[middle_high] - q50[middle_high] + 1e-8)
    return np.clip(out, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="results/ablations/quantile_cqr/config.yaml")
    parser.add_argument("--checkpoint", default="results/ablations/quantile_cqr/model_seed_42.pt")
    parser.add_argument("--output-dir", default="results/probabilistic_figures")
    parser.add_argument("--scale", choices=SCALES, default="spei_6")
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--date", help="Target date; defaults to latest date complete for all leads")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data = load_site_table(
        cfg["data"]["path"], cfg["data"].get("features"), cfg["data"]["seq_len"],
        cfg["data"]["forecast_steps"], cfg["data"].get("metadata_path"), [],
        cfg["data"].get("site_filter"), cfg["model"].get("input_extras"),
        cfg["model"].get("static_context", False), False, cfg.get("sites"))
    cfg["data"]["input_dim"] = data["test"].X.shape[-1]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(state, cfg)
    model = MultiScaleForecaster(cfg)
    model.load_state_dict(state["model"])
    model.eval()
    with torch.no_grad():
        val_output, _ = model(data["val"].X)
        test_output, _ = model(data["test"].X)
    scale_idx = SCALES.index(args.scale)
    val_q = val_output[args.scale].numpy()
    test_q = test_output[args.scale].numpy()
    val_y = data["val"].Y.numpy()[:, scale_idx]
    val_mask = data["val"].mask.numpy()[:, scale_idx]
    corrections = calibrate_cqr(val_q, val_y, val_mask, QUANTILES, coverage=0.8)
    lower, upper = apply_cqr(test_q, corrections)
    test_y = data["test"].Y.numpy()[:, scale_idx]
    test_mask = data["test"].mask.numpy()[:, scale_idx]

    rows = []
    for i, meta in enumerate(data["test"].metadata):
        for lead in range(test_q.shape[1]):
            if test_mask[i, lead] and np.isfinite(test_q[i, lead]).all():
                rows.append({
                    "site": meta["site"], "date": meta["target_dates"][lead],
                    "lead": lead + 1, "risk": float(risk_below_threshold(
                        test_q[i, lead:lead + 1], args.threshold)[0]),
                    "median": float(test_q[i, lead, 1]),
                    "lower": float(lower[i, lead]), "upper": float(upper[i, lead]),
                })
    frame = pd.DataFrame(rows)
    complete = frame.groupby("date")["lead"].nunique()
    chosen_date = args.date or str(complete.loc[complete == 4].index.max())
    snapshot = frame[frame["date"] == chosen_date]
    if snapshot.empty:
        raise ValueError(f"No complete forecast date found for {chosen_date}")

    sites = cfg["sites"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for lead, ax in enumerate(axes.flat, start=1):
        current = snapshot[snapshot["lead"] == lead].copy()
        current["lon"] = current["site"].map({k: v["lon"] for k, v in sites.items()})
        current["lat"] = current["site"].map({k: v["lat"] for k, v in sites.items()})
        sc = ax.scatter(current["lon"], current["lat"], c=current["risk"],
                        s=90 + 260 * current["risk"], cmap="Reds", vmin=.1, vmax=.9,
                        edgecolor="black", linewidth=.7)
        for _, row in current.iterrows():
            ax.annotate(f"{row.site.replace('_', ' ').title()} ({row.risk:.2f})",
                        (row.lon, row.lat), xytext=(4, 4),
                        textcoords="offset points", fontsize=7)
        ax.set_title(f"T+{lead} | {chosen_date}")
        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        ax.grid(alpha=.25)
    fig.colorbar(sc, ax=axes, label=f"Quantile-derived P(SPEI ≤ {args.threshold:.1f})")
    fig.suptitle(
        f"Probabilistic site forecast: {args.scale.upper().replace('_', '-')} drought risk\n"
        "Colors and sizes use quantile interpolation; 80% intervals are CQR-calibrated",
        y=.99)
    fig.tight_layout()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "eps"):
        fig.savefig(out / f"figure07_{args.scale}_probability_map.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    coverage = {}
    for lead in range(test_q.shape[1]):
        valid = test_mask[:, lead]
        coverage[f"T+{lead + 1}"] = interval_metrics(
            lower[valid, lead], upper[valid, lead], test_y[valid, lead])
    report = {
        "scale": args.scale, "threshold": args.threshold, "date": chosen_date,
        "checkpoint": args.checkpoint, "interval_method": "validation-only split CQR",
        "nominal_interval_coverage": 0.8,
        "test_interval_metrics": coverage,
        "probability_definition": "piecewise-linear interpolation of trained 0.1, 0.5, 0.9 quantiles",
        "probability_calibration_claim": False,
        "limitation": "CQR calibrates interval coverage, not event-probability calibration.",
    }
    (out / "probability_map_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
