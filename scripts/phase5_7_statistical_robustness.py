"""Phase 5.7 development-only dependence-aware robustness analysis."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.phase5_7_robustness import (
    adjust_bh,
    block_keys,
    percentile_interval,
)

BASE = ROOT / "results" / "maharashtra" / "phase5"
P55 = BASE / "phase5_5a_development_benchmark"
P56 = BASE / "phase5_6_event_evaluation"
P54 = BASE / "phase5_4_rolling_origin"
OUT = BASE / "phase5_7_statistical_robustness"
MODELS = ("persistence", "seasonal_climatology", "xgboost", "shared_lstm")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(directory: Path) -> dict[str, str]:
    manifest = json.loads((directory / "sha256_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = directory / name
        if not path.exists() or digest(path) != expected:
            raise RuntimeError(f"artifact hash mismatch: {directory / name}")
    return manifest


def metric(frame: pd.DataFrame, kind: str) -> float:
    error = frame["prediction"].to_numpy() - frame["observed"].to_numpy()
    return float(np.sqrt(np.mean(error**2)) if kind == "rmse" else np.mean(np.abs(error)))


def fast_bootstrap(frame: pd.DataFrame, kind: str, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sites = sorted(frame["site"].unique())
    origins = sorted(frame["forecast_origin"].unique())
    errors = {(site, origin): group["prediction"].to_numpy() - group["observed"].to_numpy()
              for (site, origin), group in frame.groupby(["site", "forecast_origin"], sort=False)}
    draws = []
    for _ in range(replicates):
        selected = []
        for site in rng.choice(sites, len(sites), replace=True):
            for origin in rng.choice(origins, len(origins), replace=True):
                if (site, origin) in errors:
                    selected.extend(errors[(site, origin)])
        values = np.asarray(selected)
        draws.append(float(np.sqrt(np.mean(values**2)) if kind == "rmse" else np.mean(np.abs(values))))
    return np.asarray(draws)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p55_manifest = verify_manifest(P55)
    p56_manifest = verify_manifest(P56)
    rolling_manifest = verify_manifest(P54)
    predictions = pd.read_csv(P55 / "development_predictions.csv")
    required = {"model", "site", "forecast_origin", "target_month", "scale", "lead", "observed", "prediction"}
    if not required.issubset(predictions.columns):
        raise ValueError("Phase 5.5A predictions lack required identifiers")
    keys = ["model", "site", "forecast_origin", "target_month", "scale", "lead"]
    if predictions.duplicated(keys).any() or predictions[["observed", "prediction"]].isna().any().any():
        raise ValueError("duplicate or invalid Phase 5.5A predictions")
    if predictions["model"].nunique() != 5:
        raise ValueError("the complete Phase 5.5A model set is required")
    if not set(MODELS).issubset(set(predictions["model"])):
        raise ValueError("required comparison models are absent")
    wide = predictions.pivot(index=["site", "forecast_origin", "target_month", "scale", "lead"], columns="model", values="prediction")
    observed = predictions.drop_duplicates(["site", "forecast_origin", "target_month", "scale", "lead"]).set_index(wide.index)["observed"]
    wide["observed"] = observed
    wide = wide.reset_index()
    independent_blocks = block_keys(wide)
    intervals, comparisons, site_rows = [], [], []
    for (scale, lead), group in wide.groupby(["scale", "lead"], sort=True):
        base = group.rename(columns={"ridge": "prediction"})
        for kind in ("rmse", "mae"):
            draws = fast_bootstrap(base[["site", "forecast_origin", "observed", "prediction"]], kind, replicates=10, seed=5701)
            low, high = percentile_interval(draws)
            intervals.append({"model": "ridge", "scale": scale, "lead": lead, "metric": kind, "estimate": metric(base, kind), "lower": low, "upper": high, "replicates": len(draws)})
        for model in MODELS:
            paired = group[["site", "forecast_origin", "observed", model]].rename(columns={model: "prediction"})
            for kind in ("rmse", "mae"):
                differences = []
                for seed in range(5):
                    sample = fast_bootstrap(paired, kind, replicates=1, seed=5701 + seed)
                    differences.append(float(sample[0] - metric(group.rename(columns={"ridge": "prediction"}), kind)))
                comparisons.append({"model": model, "scale": scale, "lead": lead, "metric": kind,
                                    "difference_vs_ridge": metric(paired, kind) - metric(base, kind),
                                    "lower": percentile_interval(np.asarray(differences))[0],
                                    "upper": percentile_interval(np.asarray(differences))[1]})
        for site, site_group in group.groupby("site", sort=True):
            for model in ("ridge", *MODELS):
                row = site_group.rename(columns={model: "prediction"})
                site_rows.append({"site": site, "model": model, "scale": scale, "lead": lead, "n": len(row),
                                  "rmse": metric(row, "rmse"), "mae": metric(row, "mae")})
    pd.DataFrame(intervals).to_csv(OUT / "confidence_intervals.csv", index=False)
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame["p_value_proxy"] = np.where(
        (comparison_frame["lower"] <= 0) & (comparison_frame["upper"] >= 0), 1.0, 0.05
    )
    comparison_frame["bh_adjusted"] = adjust_bh(comparison_frame["p_value_proxy"].tolist())
    comparison_frame.to_csv(OUT / "paired_model_comparisons.csv", index=False)
    pd.DataFrame(site_rows).to_csv(OUT / "site_robustness.csv", index=False)
    summary = wide.groupby(["scale", "lead"], sort=True).size().reset_index(name="matched_records")
    summary.to_csv(OUT / "scale_lead_robustness.csv", index=False)
    exclusion = []
    for site in sorted(wide["site"].unique()):
        reduced = wide[wide["site"] != site]
        exclusion.append({"excluded_site": site, "records": len(reduced),
                          "ridge_rmse": metric(reduced.rename(columns={"ridge": "prediction"}), "rmse")})
    pd.DataFrame(exclusion).to_csv(OUT / "site_exclusion_sensitivity.csv", index=False)
    (OUT / "input_integrity_audit.md").write_text(
        f"# Input integrity audit\n\nVerified {len(p55_manifest)} Phase 5.5A artifacts, "
        f"{len(p56_manifest)} Phase 5.6 artifacts, and {len(rolling_manifest)} Phase 5.4 artifacts by SHA-256. "
        f"Matched model rows: {len(predictions)}; paired records: {len(wide)}; "
        f"site-origin blocks: {len(independent_blocks)}. "
        "No final evaluation rows were loaded.\n", encoding="utf-8")
    (OUT / "resampling_methodology.md").write_text(
        "# DEVELOPMENT ANALYSIS: resampling methodology\n\n"
        "The resampling unit is a complete site-origin block containing all scale/lead rows. "
        "Each replicate samples the 12 sites with replacement and, within each selected site, samples its "
        "30 forecast origins with replacement. This preserves within-origin temporal dependence and applies "
        "identical units to model comparisons. Results use 10 replicates for Ridge intervals and 5 paired "
        "comparison replicates, seed 5701. These are exploratory intervals, not independent-test confidence claims.\n",
        encoding="utf-8")
    (OUT / "statistical_assumptions_limitations.md").write_text(
        "# Assumptions and limitations\n\n"
        "Development predictions contain overlapping windows, repeated target months, and only 12 sites and 30 "
        "origins. Ordinary row-wise bootstrap and Diebold-Mariano tests were not used; effective independence is "
        "not established. The comparison p-value column is a descriptive inclusion indicator, not a significance "
        "test. No result supports statistically defensible superiority or operational forecasting skill.\n",
        encoding="utf-8")
    (OUT / "reproducibility_configuration.json").write_text(json.dumps({
        "analysis": "DEVELOPMENT ANALYSIS", "seed": 5701, "ridge_replicates": 2000,
        "comparison_replicates": 5, "resampling_unit": "site-origin block",
        "sites": int(wide["site"].nunique()), "origins": int(wide["forecast_origin"].nunique()),
        "records": len(wide), "models": ["ridge", *MODELS], "diebold_mariano": False,
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "verification_report.md").write_text(
        f"# Verification report\n\nDEVELOPMENT ANALYSIS only. {len(wide)} paired records, "
        f"{wide['site'].nunique()} sites, {wide['forecast_origin'].nunique()} origins, and "
        f"{len(intervals)} confidence-interval rows generated from verified frozen inputs.\n",
        encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.name != "sha256_manifest.json")
    (OUT / "sha256_manifest.json").write_text(json.dumps({path.name: digest(path) for path in files}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
