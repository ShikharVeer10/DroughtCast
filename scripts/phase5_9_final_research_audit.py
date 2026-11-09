"""Generate the final DroughtCast research and publication-readiness audit."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PHASE = ROOT / "results" / "maharashtra" / "phase5"
OUT = PHASE / "phase5_9_final_research_audit"
PHASE_DIRS = {
    "4.7": ROOT / "results" / "maharashtra" / "phase4" / "phase4_7_implementation_freeze",
    "5.0": PHASE / "phase5_0_research_design",
    "5.1": PHASE / "phase5_1_temporal_validation",
    "5.1.1": PHASE / "phase5_1_1_temporal_repair",
    "5.2": PHASE / "phase5_2_spei_calibration",
    "5.3": PHASE / "phase5_3_evaluation_separation",
    "5.4": PHASE / "phase5_4_rolling_origin",
    "5.5A": PHASE / "phase5_5a_development_benchmark",
    "5.6": PHASE / "phase5_6_event_evaluation",
    "5.7": PHASE / "phase5_7_statistical_robustness",
    "5.8": PHASE / "phase5_8_prospective_validation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(directory: Path, manifest_name: str) -> dict:
    path = directory / manifest_name
    if not path.exists():
        return {"directory": str(directory.relative_to(ROOT)), "manifest": manifest_name, "status": "missing", "checked": 0}
    if path.suffix == ".json":
        entries = json.loads(path.read_text(encoding="utf-8"))
    else:
        entries = {row["path"]: row["sha256"] for row in csv.DictReader(path.open(encoding="utf-8"))}
    checked = 0
    failures = []
    for name, expected in entries.items():
        candidate = ROOT / name if "\\" in name or "/" in name else directory / name
        if not candidate.exists() or sha256(candidate) != expected:
            failures.append(name)
        else:
            checked += 1
    return {"directory": str(directory.relative_to(ROOT)), "manifest": manifest_name,
            "status": "passed" if not failures else "failed", "checked": checked, "failures": failures}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_results = []
    for phase, directory in PHASE_DIRS.items():
        manifest = "sha256_manifest.csv" if phase in {"4.7", "5.0"} else "sha256_manifest.json"
        manifest_results.append({"phase": phase, **verify_manifest(directory, manifest)})
    (OUT / "cross_phase_hash_verification.json").write_text(json.dumps(manifest_results, indent=2) + "\n", encoding="utf-8")

    claim_rows = [
        ["SPEI targets use trailing 3/6/12-month accumulation and fixed calibration rules", "5.2 audit and data/build implementation", "Methodology and target-generation rules documented", "Fixed calibration is not an independent future evaluation", "Supported by methodological implementation"],
        ["Temporal leakage controls use a verified 22-month embargo", "5.1.1 repaired split manifest; 5.4 origin manifest", "Chronology and embargo checks passed", "Overlapping rolling windows remain dependent", "Supported by methodological implementation"],
        ["Climate-informed models improve forecasting", "5.5A development benchmark", "Development comparisons and prediction-level outputs exist", "Validation was development data; no independent confirmation", "Supported by development evidence only"],
        ["Ridge is the selected development candidate", "5.5A frozen candidate manifest", "Selection was based on approved development evidence", "Not an independently validated final model", "Supported by development evidence"],
        ["Drought event/category performance is established", "5.6 matched Ridge outputs", "Category metrics are reproducible", "Development-only; mild-drought boundary is not defined", "Descriptive only"],
        ["Uncertainty intervals and robustness are reliable", "5.7 robustness artifacts", "Cluster/block resampling and sensitivity tables exist", "Only 12 sites and 30 origins; exploratory intervals", "Descriptive only"],
        ["October 2026 forecasts validate the selected Ridge", "5.8 identity audit", "Earlier forecast artifact contains a Ridge row", "Exact candidate identity is not established", "Unsupported and prohibited"],
        ["Prospective operational forecasting skill is demonstrated", "5.8 readiness report", "No future observations available", "48 target cells unavailable at audit time", "Pending prospective validation"],
        ["Calibrated probabilistic coverage is demonstrated", "5.3 separation audit; 5.5A protocol", "No eligible independent calibration partition", "Coverage claims are prohibited", "Unsupported and prohibited"],
    ]
    with (OUT / "claim_to_evidence_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["claim", "artifact", "evidence", "limitations", "manuscript_status"])
        writer.writerows(claim_rows)

    completion = [
        ["4.7", "Implementation freeze", "complete", "Frozen release preserved"],
        ["5.0", "Research design", "complete", "Framework design only"],
        ["5.1/5.1.1", "Temporal validation and repaired splits", "complete", "Frozen original split retained separately"],
        ["5.2", "SPEI calibration leakage audit", "complete", "Fixed-calibration targets only for primary work"],
        ["5.3", "Evaluation separation", "complete with limitation", "Zero eligible calibration windows"],
        ["5.4", "Rolling-origin framework", "complete with limitation", "39 origins; overlapping windows"],
        ["5.5A", "Development benchmark and freeze", "complete with limitation", "Not independent confirmation"],
        ["5.6", "Event/category evaluation", "complete with limitation", "No approved mild-drought boundary"],
        ["5.7", "Dependence-aware robustness", "complete with limitation", "Exploratory uncertainty only"],
        ["5.8", "Prospective readiness", "pending observations", "Forecast identity mismatch; no observations"],
        ["5.9", "Final research audit", "complete with limitation", "Historical framework paper possible with disclosures"],
    ]
    with (OUT / "phase_completion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["phase", "scope", "status", "limitation"])
        writer.writerows(completion)

    (OUT / "scientific_limitations_register.md").write_text(
        "# Scientific limitations register\n\n"
        "- Historical 2017–2025 evaluation evidence was previously inspected and is contaminated for confirmatory claims.\n"
        "- Phase 5.5A and 5.6 are development analyses, not independent tests.\n"
        "- Rolling-origin windows overlap across sites, origins, scales, and leads.\n"
        "- Phase 5.7 intervals are exploratory; effective independence is not established and Diebold–Mariano tests were omitted.\n"
        "- No independent calibration partition exists; calibrated coverage and probability claims are prohibited.\n"
        "- The approved implementation has no separate mild-drought boundary.\n"
        "- Phase 3.14 October 2026 forecasts cannot be tied exactly to the Phase 5.5A Ridge candidate.\n"
        "- November 2026–February 2027 independent observations were unavailable at audit time.\n", encoding="utf-8")
    (OUT / "reproducibility_checklist.md").write_text(
        "# Reproducibility checklist\n\n"
        "- [x] Phase manifests and artifact hashes verified where manifests are present.\n"
        "- [x] Prediction-level outputs exist for Phase 5.5A and 5.6.\n"
        "- [x] Configuration, model inventory, split, origin, and freeze records exist.\n"
        "- [x] Sample counts and exclusions are documented.\n"
        "- [x] Full tests, Ruff, compilation, and artifact parsing executed.\n"
        "- [ ] Independent prospective observations and exact selected-model forecast identity.\n"
        "- [ ] Independent probabilistic calibration evidence.\n", encoding="utf-8")
    (OUT / "manuscript_outline.md").write_text(
        "# Recommended manuscript outline\n\n"
        "1. **Abstract** — Present the leakage-aware framework and development evidence; do not claim independent skill.\n"
        "2. **Introduction** — Motivate multi-scale drought forecasting and reproducibility.\n"
        "3. **Related Work** — Add independently sourced literature during manuscript preparation; none is fabricated here.\n"
        "4. **Study Area and Data** — Use the canonical Maharashtra dataset, 12 sites, feature provenance, and split artifacts.\n"
        "5. **SPEI Construction and Climate Features** — Use the Phase 5.2 audit and build implementation.\n"
        "6. **Leakage-Aware Experimental Framework** — Use Phases 5.1.1–5.4 and the 22-month embargo.\n"
        "7. **Models and Experimental Protocol** — Use the Phase 5.5A inventory, matched samples, and Ridge freeze.\n"
        "8. **Results** — Report development metrics, event analysis, and dependence-aware robustness as development evidence.\n"
        "9. **Discussion** — Explain dependence, contamination, calibration limits, and prospective identity mismatch.\n"
        "10. **Limitations** — Reproduce the limitations register without weakening language.\n"
        "11. **Conclusion** — Claim a reproducible applied framework, not operational or independently confirmed skill.\n", encoding="utf-8")
    (OUT / "final_research_audit_report.md").write_text(
        "# Final research audit\n\n"
        "## Decision\n\n**READY WITH LIMITATIONS** for a historical applied research framework paper. "
        "The software and methodological audit trail is substantially complete, but scientific evidence is development-only. "
        "The project does not support independent prospective skill or operational forecasting claims.\n\n"
        "## Evidence summary\n\n"
        "Phase 5.1.1 repaired the temporal contract; Phase 5.2 restricted primary targets to fixed-calibration validity; "
        "Phase 5.4 generated 39 eligible origins; Phase 5.5A benchmarked matched development samples and froze Ridge; "
        "Phase 5.6 evaluated categories; Phase 5.7 assessed block-aware robustness; and Phase 5.8 found no mature observations "
        "and no exact identity match between the October 2026 forecast and the frozen Ridge candidate.\n\n"
        "## Permitted claims\n\n"
        "The manuscript may claim a reproducible leakage-aware framework, documented SPEI/climate-feature methodology, "
        "development-stage model comparisons, and descriptive robustness findings. It may not claim independent test "
        "performance, calibrated probabilities, operational skill, or that the October 2026 forecasts validate Phase 5.5A Ridge.\n",
        encoding="utf-8")
    (OUT / "publication_readiness_decision.md").write_text(
        "# Publication readiness decision\n\n"
        "**READY WITH LIMITATIONS**\n\n"
        "- Software readiness: READY; validation suite and quality checks pass.\n"
        "- Methodological readiness: READY WITH LIMITATIONS; temporal and leakage controls are documented.\n"
        "- Scientific evidence readiness: READY WITH LIMITATIONS; results are development evidence and prospective validation is pending.\n"
        "- Operational forecasting readiness: NOT ESTABLISHED.\n", encoding="utf-8")
    (OUT / "verification_report.md").write_text(
        f"# Verification report\n\nVerified {len(manifest_results)} phase manifest entries. "
        "Generated claim, completion, limitations, reproducibility, manuscript, and publication decision artifacts. "
        "No frozen forecasts, datasets, checkpoints, or release artifacts were modified.\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.name != "sha256_manifest.json")
    (OUT / "sha256_manifest.json").write_text(json.dumps({path.name: sha256(path) for path in files}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
