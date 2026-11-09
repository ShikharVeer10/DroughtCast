import json

from scripts.phase5_8_prospective_validation import P55, P314


def test_frozen_forecast_and_candidate_are_not_claimed_identical():
    metadata = json.loads((P314 / "forecast_metadata.json").read_text(encoding="utf-8"))
    candidate = json.loads((P55 / "frozen_candidate_manifest.json").read_text(encoding="utf-8"))
    assert "ridge" in metadata["models"]
    assert candidate["selected_model"] == "ridge"
    assert candidate["prospective_outcome_change_rule"]


def test_future_observations_are_not_present_in_source_manifest():
    metadata = json.loads((P314 / "forecast_metadata.json").read_text(encoding="utf-8"))
    assert metadata["future_targets_used"] is False
