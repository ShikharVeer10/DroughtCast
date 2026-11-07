"""Regenerate the readiness ledger from required paper evidence artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_EVIDENCE = {
    "local_target_diagnostics": "results/maharashtra/protocol7_final_local/protocol_diagnostics.json",
    "speibase_target_diagnostics": "results/maharashtra/protocol7_final_speibase/protocol_diagnostics.json",
    "paired_target_comparison": "results/maharashtra/protocol7_final_target_comparison/target_source_comparison.csv",
    "independent_pet_sensitivity": "results/maharashtra/pet_sensitivity/pet_comparison.csv",
    "distribution_sensitivity": "results/maharashtra/distribution_sensitivity/summary.csv",
    "five_seed_local_test_predictions": "results/maharashtra/local_targets/seeds/summary.csv",
    "five_seed_speibase_test_predictions": "results/maharashtra/speibase_targets/seeds/summary.csv",
    "seq_len_ablation": "results/maharashtra/ablations/seq_len/summary.csv",
    "region_holdouts": "results/maharashtra/region_holdouts/summary.csv",
    "probability_test_calibration": "results/maharashtra/probability_calibration/test_metrics.json",
    "environment_manifest": "results/environment.json",
    "paper_self_review": "paper/SELF_REVIEW.md",
}


def build_readiness() -> dict:
    evidence = {name: (ROOT / path).is_file() for name, path in REQUIRED_EVIDENCE.items()}
    missing = [name for name, exists in evidence.items() if not exists]
    diagnostics_paths = [
        ROOT / "results/maharashtra/protocol7_final_local/protocol_diagnostics.json",
        ROOT / "results/maharashtra/protocol7_final_speibase/protocol_diagnostics.json",
    ]
    diagnostic_warnings = {}
    for path in diagnostics_paths:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            diagnostic_warnings[str(path.relative_to(ROOT))] = data.get("warnings", [])
    unresolved = []
    for path, warnings in diagnostic_warnings.items():
        if warnings:
            unresolved.append(f"Target diagnostics have unresolved warnings: {path}")
    status = "publication_ready" if not missing and not unresolved else "not_publication_ready"
    return {
        "status": status,
        "generated_by": "python -m experiments.update_research_readiness",
        "evidence": evidence,
        "missing_evidence": missing,
        "diagnostic_warnings": diagnostic_warnings,
        "unresolved_blockers": unresolved,
        "policy": "Publication readiness requires every listed artifact and zero unresolved blocking diagnostic warnings.",
    }


def main() -> None:
    output = ROOT / "docs/research_readiness.json"
    readiness = build_readiness()
    output.write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    print(json.dumps({"status": readiness["status"], "missing_evidence": readiness["missing_evidence"]}, indent=2))


if __name__ == "__main__":
    main()
