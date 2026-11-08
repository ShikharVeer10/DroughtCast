import pandas as pd

from data.availability_monitor import (
    POWER_VARIABLES,
    inspect_climate_indices,
    inspect_power_frame,
    validate_candidate_frame,
)


def test_mocked_power_response_detects_missing_variable_and_invalid_value():
    frame = pd.DataFrame({"date": ["2026-11-01"], **{v: [1.0] for v in POWER_VARIABLES[:-1]}})
    frame["T2M_RANGE"] = float("nan")
    result = inspect_power_frame("akola", frame)
    assert result.complete is False
    assert "VPD" in result.missing_variables


def test_stale_index_is_not_usable():
    rows = inspect_climate_indices({"oni": "2026-01", "dmi": None, "nao": "2026-10", "pdo": "2026-03"}, "2026-11")
    assert next(row for row in rows if row["index"] == "dmi")["usable"] is False
    assert next(row for row in rows if row["index"] == "oni")["usable"] is False


def test_duplicate_candidate_cannot_pass():
    frame = pd.DataFrame({"date": ["2026-11-01", "2026-11-01"], "site": ["akola", "akola"]})
    assert "duplicate_site_month" in validate_candidate_frame(frame, {"akola"})
