"""Audit frozen prospective forecasts without consuming future observations."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

P314 = ROOT / "results" / "maharashtra" / "phase3" / "phase3_14_prospective_forecasts"
P47 = ROOT / "results" / "maharashtra" / "phase4" / "phase4_7_implementation_freeze"
P55 = ROOT / "results" / "maharashtra" / "phase5" / "phase5_5a_development_benchmark"
OUT = ROOT / "results" / "maharashtra" / "phase5" / "phase5_8_prospective_validation"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((P314 / "forecast_metadata.json").read_text(encoding="utf-8"))
    release = json.loads((P47 / "release_manifest.json").read_text(encoding="utf-8"))
    candidate = json.loads((P55 / "frozen_candidate_manifest.json").read_text(encoding="utf-8"))
    ridge_config = json.loads((P55 / "selected_model_config.json").read_text(encoding="utf-8"))
    forecast_files = ["classical_forecasts.csv", "forecast_summary.csv", "neural_forecasts.csv", "site_level_forecasts.csv"]
    forecast_hashes = {name: sha256(P314 / name) for name in forecast_files}
    identity = {
        "forecast_origin": metadata["forecast_origin"],
        "target_months": metadata["target_months"],
        "sites": metadata["sites"],
        "features": metadata["features"],
        "normalizer_sha256": metadata["normalizer_sha256"],
        "dataset_sha256": metadata["dataset_sha256"],
        "phase314_models": metadata["models"],
        "phase55_selected_model": candidate["selected_model"],
        "phase55_selected_config": ridge_config,
        "phase47_revision": release["repository_revision"],
        "phase314_forecast_hashes": forecast_hashes,
        "exact_ridge_identity_established": False,
        "classification": "separate earlier prospective forecast experiment",
        "reason": "Phase 3.14 contains a Ridge row and shared dataset/normalizer hashes, but no hash or configuration identity tying that Ridge artifact to the Phase 5.5A frozen candidate.",
    }
    (OUT / "frozen_forecast_identity_audit.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    (OUT / "ridge_compatibility_report.md").write_text(
        "# Ridge compatibility report\n\n"
        "The October 2026 Phase 3.14 release includes a Ridge forecast and uses the same canonical dataset, "
        "Protocol 10 normalizer, feature list, origin, sites, scales, and leads recorded in the later manifests. "
        "However, Phase 3.14 does not record the Phase 5.5A selected-candidate configuration or a matching model "
        "artifact hash. Exact identity is therefore **not established**. The forecast is retained as a separate "
        "earlier prospective experiment and must not be presented as validation of the Phase 5.5A Ridge candidate.\n",
        encoding="utf-8")
    sites = metadata["sites"]
    targets = metadata["target_months"]
    rows = []
    for target in targets:
        for site in sites:
            rows.append({"target_month": target, "site": site, "status": "unavailable",
                         "observation_available": False, "source": "", "acquisition_date": "",
                         "observation_sha256": "", "methodology": "not acquired; no target observations loaded"})
    pd.DataFrame(rows).to_csv(OUT / "observation_availability_matrix.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "observation_provenance.csv", index=False)
    (OUT / "prediction_observation_matching_report.md").write_text(
        "# Prediction-observation matching report\n\n"
        "Matching was not performed. As of the audit date (2026-11-09), the four-month target horizon "
        "November 2026 through February 2027 has not matured and no independent observation artifact was loaded. "
        "No metrics or category scores were calculated.\n", encoding="utf-8")
    (OUT / "limitations_and_readiness_report.md").write_text(
        "# Limitations and readiness\n\n"
        "**Status: PENDING OBSERVATIONS.** The protocol is ready for a future independent observation freeze, "
        "but observations are unavailable. SPEIbase is not treated as ground truth. The Phase 3.14 forecast "
        "identity is not exactly compatible with the Phase 5.5A Ridge candidate, so it remains a separate earlier "
        "prospective experiment. No retraining, retuning, or forecast replacement is permitted after outcomes mature.\n",
        encoding="utf-8")
    (OUT / "verification_report.md").write_text(
        f"# Verification report\n\nFrozen forecast files hashed: {len(forecast_hashes)}. "
        f"Observation cells audited: {len(rows)}. Observation cells available: 0. "
        f"Audit date: {date.today().isoformat()}. No future observations were loaded.\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.name != "sha256_manifest.json")
    (OUT / "sha256_manifest.json").write_text(json.dumps({path.name: sha256(path) for path in files}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
