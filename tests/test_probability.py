import numpy as np
import pytest

from utils.probability import (brier_score, expected_calibration_error,
                               fit_isotonic_calibrator)


def test_isotonic_calibration_is_bounded_and_finite():
    calibrator = fit_isotonic_calibrator(
        np.array([.1, .2, .7, .9]), np.array([0, 0, 1, 1]))
    result = calibrator.predict([.0, .5, 1.0])
    assert np.isfinite(result).all()
    assert ((result >= 0) & (result <= 1)).all()


def test_probability_metrics_validate_empty_input():
    assert np.isnan(brier_score(np.array([np.nan]), np.array([1])))
    assert np.isnan(expected_calibration_error(np.array([np.nan]), np.array([1])))
    with pytest.raises(ValueError):
        fit_isotonic_calibrator(np.array([.5, .5]), np.array([1, 1]))
