import pytest

from data.temporal_validation import TemporalRequirements, validate_samples

BOUNDARIES = {
    "train_end": "2010-12-01", "val_start": "2012-10-01",
    "val_end": "2015-12-01", "test_start": "2017-10-01", "test_end": "2025-12-01",
}
REQ = TemporalRequirements(6, 4)


def sample(**changes):
    value = {"site": "a", "split": "train",
             "input_dates": ["2010-06-01", "2010-07-01"],
             "target_dates": ["2010-08-01", "2010-09-01"], "target": "spei_3"}
    value.update(changes)
    return value


def test_overlapping_target_windows_fail():
    result = validate_samples([sample(), sample(input_dates=["2010-07-01", "2010-08-01"],
                                                 target_dates=["2010-08-01", "2010-09-01"])],
                              BOUNDARIES, REQ)
    assert any("overlap" in item["reason"] for item in result["violations"])


def test_duplicate_origin_fails():
    result = validate_samples([sample(), sample(target="spei_6")], BOUNDARIES, REQ)
    assert result["duplicate_forecast_origins"]


def test_invalid_order_and_future_split_fail():
    with pytest.raises(ValueError):
        validate_samples([sample(split="val")], {**BOUNDARIES, "val_start": "2010-01-01"}, REQ)
    result = validate_samples([sample(split="train", target_dates=["2011-01-01"])], BOUNDARIES, REQ)
    assert any("boundary" in item["reason"] for item in result["violations"])


def test_missing_site_month_target_and_embargo_are_reported():
    result = validate_samples([sample(target_values=[-1.0, None])], BOUNDARIES, REQ)
    assert result["duplicate_site_month_target"] == []
    assert any("missing site-month target" in item["reason"] for item in result["violations"])
    assert result["embargo"]["train_to_val"]["pass"]


def test_insufficient_embargo_is_rejected():
    original = {**BOUNDARIES, "val_start": "2012-01-01", "test_start": "2017-01-01"}
    result = validate_samples([], original, REQ)
    assert not result["embargo"]["train_to_val"]["pass"]
    assert not result["embargo"]["val_to_test"]["pass"]
