"""Evaluate Ridge development drought categories without loading final evaluation data."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.phase5_6_events import CATEGORIES, THRESHOLDS, classification_metrics

BASE = ROOT / "results" / "maharashtra" / "phase5" / "phase5_5a_development_benchmark"
OUT = ROOT / "results" / "maharashtra" / "phase5" / "phase5_6_event_evaluation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BASE / "sha256_manifest.json").read_text(encoding="utf-8"))
    prediction_path = BASE / "development_predictions.csv"
    if manifest["development_predictions.csv"] != sha256(prediction_path):
        raise RuntimeError("Phase 5.5A prediction hash mismatch")
    frame = pd.read_csv(prediction_path, parse_dates=["forecast_origin", "target_month"])
    frame = frame[frame["model"].eq("ridge")].copy()
    required = {"site", "forecast_origin", "target_month", "scale", "lead", "observed", "prediction"}
    if not required.issubset(frame.columns) or frame.duplicated(["site", "forecast_origin", "target_month", "scale", "lead"]).any():
        raise ValueError("Prediction output is missing required fields or contains duplicates")
    if frame[["observed", "prediction"]].isna().any().any():
        raise ValueError("Missing values in matched Ridge predictions")
    frame.to_csv(OUT / "matched_ridge_predictions.csv", index=False)
    metric_rows, category_rows, confusion_rows, site_rows = [], [], [], []
    for (scale, lead), group in frame.groupby(["scale", "lead"], sort=True):
        matrix, summary = classification_metrics(group)
        metric_rows.append({"scale": scale, "lead": lead, "n": len(group), **{k: v for k, v in summary[0].items() if k != "per_category"}})
        category_rows.extend({"scale": scale, "lead": lead, **row} for row in summary[0]["per_category"])
        confusion_rows.append({"scale": scale, "lead": lead, "matrix": matrix.tolist()})
        for site, site_group in group.groupby("site", sort=True):
            _, site_summary = classification_metrics(site_group)
            site_rows.append({"site": site, "scale": scale, "lead": lead, "n": len(site_group), **{k: v for k, v in site_summary[0].items() if k != "per_category"}})
    pd.DataFrame(metric_rows).to_csv(OUT / "per_scale_lead_metrics.csv", index=False)
    pd.DataFrame(category_rows).to_csv(OUT / "per_category_metrics.csv", index=False)
    pd.DataFrame(site_rows).to_csv(OUT / "site_metrics.csv", index=False)
    (OUT / "confusion_matrices.json").write_text(json.dumps(confusion_rows, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    category_counts = []
    for (scale, lead), group in frame.groupby(["scale", "lead"], sort=True):
        for kind, values in (("observed", group["observed"]), ("predicted", group["prediction"])):
            counts = pd.Series(np.bincount(np.digitize(values.to_numpy(), THRESHOLDS, right=True), minlength=5))
            for index, name in enumerate(CATEGORIES):
                category_counts.append({"scale": scale, "lead": lead, "kind": kind, "category": name, "count": int(counts.iloc[index])})
    pd.DataFrame(category_counts).to_csv(OUT / "category_counts.csv", index=False)
    # Event continuity is assessed on one earliest-origin forecast per site/target month/scale.
    event_frame = frame.sort_values("forecast_origin").drop_duplicates(["site", "target_month", "scale"])
    event_rows = []
    for (site, scale), group in event_frame.groupby(["site", "scale"], sort=True):
        observed = group["observed"] <= -1.0
        predicted = group["prediction"] <= -1.0
        event_rows.append({"site": site, "scale": scale, "observed_drought_observations": int(observed.sum()),
                           "predicted_drought_observations": int(predicted.sum()),
                           "matched_drought_observations": int((observed & predicted).sum()),
                           "event_rule": "earliest forecast per site-target-month-scale; SPEI <= -1.0"})
    pd.DataFrame(event_rows).to_csv(OUT / "event_analysis.csv", index=False)
    (OUT / "evaluation_configuration.json").write_text(json.dumps({
        "model": "ridge", "source_hash": manifest["development_predictions.csv"],
        "scales": ["spei_3", "spei_6", "spei_12"], "leads": [1, 2, 3, 4],
        "thresholds": {"extreme": "<= -2.0", "severe": "(-2.0, -1.5]", "moderate": "(-1.5, -1.0]", "non_drought": "(-1.0, 1.0]", "wet": "> 1.0"},
        "final_evaluation_loaded": False, "independent_evaluation": False,
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "exclusion_report.md").write_text(
        f"# Exclusion and sample accounting\n\nMatched Ridge records: **{len(frame)}**. "
        "Duplicates, missing values, and hash mismatches are rejected. The source contains 360 "
        "records per scale/lead across 12 sites and 30 development origins. No final-evaluation rows were loaded.\n",
        encoding="utf-8")
    (OUT / "limitations_report.md").write_text(
        "# Limitations\n\n"
        "These are development-stage event results only. The repository has no approved separate mild-drought "
        "boundary; the existing thresholds distinguish moderate, severe, and extreme drought, with non-drought "
        "and wet classes. Mild drought is therefore not separately reported. Event rows use an earliest-origin "
        "deduplication and are not independent. No calibrated probabilities, independent performance, or significance is claimed.\n",
        encoding="utf-8")
    (OUT / "verification_report.md").write_text(
        f"# Verification report\n\nRidge records: {len(frame)}. Metrics: {len(metric_rows)} scale/lead cells. "
        "Source SHA-256 verified; final evaluation and frozen prospective artifacts were not loaded.\n",
        encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.name != "sha256_manifest.json")
    (OUT / "sha256_manifest.json").write_text(json.dumps({path.name: sha256(path) for path in files}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
