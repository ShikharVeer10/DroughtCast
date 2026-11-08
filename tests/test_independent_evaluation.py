import numpy as np
import pytest

from evaluation.independent import (
    IndependentEvaluationSpec,
    classification_metrics,
    validate_candidate_windows,
)


def spec():
    return IndependentEvaluationSpec(
        sites=tuple(f"site{i}" for i in range(12)),
        features=tuple(f"feature{i}" for i in range(15)),
    )


def test_independent_windows_are_six_plus_four_and_nonoverlapping():
    windows = spec().windows()
    assert windows[0] == {
        "origin": "2026-06",
        "input_start": "2026-01",
        "input_end": "2026-06",
        "target_start": "2026-07",
        "target_end": "2026-10",
    }
    assert windows[-1]["target_end"] == "2027-02"


def test_invalid_origin_and_consumed_overlap_rejected():
    with pytest.raises(ValueError):
        IndependentEvaluationSpec(
            origins=("2025-12",) * 5,
            sites=spec().sites,
            features=spec().features,
        ).validate()


def test_candidate_windows_require_all_origins():
    with pytest.raises(ValueError):
        validate_candidate_windows(spec().windows()[:-1], spec())


def test_classification_metrics_keep_undefined_as_nan():
    result = classification_metrics(np.array([0.0, 0.1]), np.array([0.0, 0.1]))
    assert np.isnan(result["recall"])
    assert np.isnan(result["precision"])
