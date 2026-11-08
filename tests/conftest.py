from pathlib import Path

import pytest


# Tests below validate repository-local artifacts that are intentionally gitignored
# in lightweight CI clones. They should run only when those artifacts are present.
_ARTIFACT_DEPENDENT_TESTS = {
    "tests/test_phase2_1_baselines.py::test_verified_normalizer_metadata_is_accepted_and_legacy_is_rejected": [
        "data/meta.json",
    ],
    "tests/test_phase2_4_2_fixes.py::test_historical_checkpoint_configuration_keeps_gate_disabled": [
        "results/maharashtra/phase2/neural/protocol10_seed42_e9dfa7ce76_ae290b4115_1d7a5ce0/neural_experiment_metadata.json",
    ],
    "tests/test_phase2_4_neural_audit.py::test_model_head_scale_lead_quantile_mapping_and_gradient_flow": [
        "results/maharashtra/phase2/neural/protocol10_seed42_e9dfa7ce76_ae290b4115_1d7a5ce0/neural_experiment_metadata.json",
    ],
    "tests/test_phase311_readiness_gate.py::test_incomplete_future_data_blocks_evaluation": [
        "data/maharashtra/monthly_sites.csv",
    ],
    "tests/test_phase312_forecast_gate.py::test_november_to_february_forecast_origin_has_incomplete_features": [
        "data/maharashtra/monthly_sites.csv",
    ],
    "tests/test_phase314_forecasts.py::test_prospective_forecast_outputs_have_expected_keys_and_finite_values": [
        "results/maharashtra/phase3/phase3_14_prospective_forecasts/classical_forecasts.csv",
        "results/maharashtra/phase3/phase3_14_prospective_forecasts/neural_forecasts.csv",
    ],
    "tests/test_phase315_forecast_analysis.py::test_forecast_analysis_outputs_match_phase314_records": [
        "results/maharashtra/phase3/phase3_15_forecast_analysis/location_wise_summary.csv",
        "results/maharashtra/phase3/phase3_15_forecast_analysis/scale_comparison.csv",
        "results/maharashtra/phase3/phase3_15_forecast_analysis/model_comparison.csv",
    ],
    "tests/test_phase316_reproducibility.py::test_phase316_hash_audit_reports_no_failures": [
        "results/maharashtra/phase3/phase3_16_reproducibility_audit/artifact_hash_verification.json",
    ],
    "tests/test_phase39_data_freeze.py::test_required_future_months_are_not_present_in_canonical_data": [
        "data/maharashtra/monthly_sites.csv",
    ],
    "tests/test_preprocessing_integrity.py::test_october_2026_inference_preprocessing_imputes_stale_indices": [
        "data/maharashtra/monthly_sites.csv",
        "data/maharashtra/dataset_metadata.json",
    ],
    "tests/test_spei_phase1_3_audit.py::test_beed_september_2019_reconstruction_matches_canonical": [
        "data/maharashtra/monthly_sites.csv",
        "results/maharashtra/speibase_validation/matched_monthly_points.csv",
    ],
    "tests/test_spei_phase1_3_audit.py::test_dated_nonparametric_1990_alternative_reproduces_all_canonical_targets": [
        "data/maharashtra/monthly_sites.csv",
        "results/maharashtra/phase1/spei_alternative_monthly.csv",
    ],
    "tests/test_spei_phase1_4_analysis.py::test_beed_september_2019_top_discrepancy_reconstruction_is_reproducible": [
        "data/maharashtra/monthly_sites.csv",
        "results/maharashtra/phase1/spei_extreme_event_audit.csv",
    ],
    "tests/test_spei_phase1_4_analysis.py::test_phase14_run_preserves_canonical_and_config": [
        "data/maharashtra/monthly_sites.csv",
        "results/maharashtra/phase1/spei_extreme_event_audit.csv",
        "results/maharashtra/phase1/spei_alternative_monthly.csv",
        "results/maharashtra/phase1/spei_fit_diagnostics.json",
    ],
}


def pytest_collection_modifyitems(config, items):
    repo_root = Path(__file__).resolve().parents[1]
    for item in items:
        required = _ARTIFACT_DEPENDENT_TESTS.get(item.nodeid)
        if not required:
            continue
        missing = [rel for rel in required if not (repo_root / rel).exists()]
        if missing:
            reason = "requires local artifacts not present in this checkout: " + ", ".join(missing)
            item.add_marker(pytest.mark.skip(reason=reason))
