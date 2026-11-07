import numpy as np
import pandas as pd
import pytest

from experiments.validate_spei_reference import (
    _event_agreement,
    align_monthly,
    compare_pair,
    drought_category,
    haversine_km,
    nearest_grid_cell,
)


def test_monthly_alignment_does_not_interpolate_missing_reference_month():
    local = pd.DataFrame({"site": ["pune", "pune", "pune"], "date": pd.to_datetime(["2000-01-15", "2000-02-15", "2000-03-15"]), "local": [0.1, 0.2, 0.3]})
    reference = pd.DataFrame({"site": ["PUNE", "PUNE"], "date": pd.to_datetime(["2000-01-01", "2000-03-01"]), "reference": [0.0, 0.4]})

    aligned = align_monthly(local, reference, "local", "reference")

    assert len(aligned) == 3
    assert pd.isna(aligned.loc[1, "reference"])
    assert aligned.loc[1, "date"] == pd.Timestamp("2000-02-01")


def test_alignment_rejects_duplicate_site_months():
    frame = pd.DataFrame({"site": ["pune", "pune"], "date": pd.to_datetime(["2000-01-02", "2000-01-28"]), "v": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Duplicate site/month"):
        align_monthly(frame, frame, "v", "v")


def test_nearest_reference_cell_uses_great_circle_distance():
    cells = [(20.75, 77.25), (18.75, 73.75)]
    assert nearest_grid_cell(18.5204, 73.8567, cells) == (18.75, 73.75)
    assert haversine_km(18.5204, 73.8567, 18.75, 73.75) > 0


def test_drought_category_thresholds_and_missing_values():
    categories = drought_category(np.array([0.0, -1.0, -1.5, -2.0, np.nan]))
    np.testing.assert_array_equal(categories, [0, 1, 2, 3, -1])


def test_event_onset_and_termination_agreement_uses_month_tolerance():
    frame = pd.DataFrame({
        "date": pd.date_range("2000-01-01", periods=6, freq="MS"),
        "local": [0.0, -1.6, -1.7, 0.0, 0.0, 0.0],
        "reference": [0.0, 0.0, -1.6, -1.7, 0.0, 0.0],
    })

    result = _event_agreement(frame, "local", "reference", -1.5)

    assert result["local_event_count"] == 1
    assert result["reference_event_count"] == 1
    assert result["matched_events_within_tolerance"] == 1
    assert result["mean_absolute_onset_error_months"] == 1.0
    assert result["mean_absolute_termination_error_months"] == 1.0


def test_comparison_sample_count_and_four_class_confusion_matrix():
    aligned = pd.DataFrame({
        "site": ["pune"] * 4,
        "date": pd.date_range("2000-01-01", periods=4, freq="MS"),
        "spei_3_local": [0.2, -1.2, -1.7, -2.1],
        "spei_3_reference": [0.1, -1.6, -1.2, -2.1],
    })

    metrics, detail = compare_pair(
        "pune", 3, "synthetic", aligned,
        {"lat": 18.5, "lon": 73.8},
        {"reference_lat": 18.75, "reference_lon": 73.75},
        "2000-01-01", "2000-04-01",
    )

    assert metrics["n_pairs"] == 4
    assert detail["category_metrics"]["four_class_confusion_matrix_reference_rows_local_columns"] == [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
