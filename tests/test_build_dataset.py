import numpy as np

from data.build_dataset import _nonparametric_z_scores


def test_nonparametric_extreme_tails_are_finite_and_clipped():
    calibration = np.linspace(-3, 3, 25)
    result = _nonparametric_z_scores(np.array([-20, 20]), calibration)
    assert np.isfinite(result).all()
    assert np.all(np.abs(result) <= 3.5)


def test_nonparametric_recovery_calibration_is_increasing():
    calibration = np.linspace(2, 4, 20)
    result = _nonparametric_z_scores(np.array([-10, 10]), calibration)
    assert result[0] < result[1]


def test_nonparametric_constant_calibration_is_finite_and_clipped():
    result = _nonparametric_z_scores(np.array([-5, 9]), np.array([2] * 5))
    assert np.isfinite(result).all()
    assert np.all(np.abs(result) <= 3.5)


def test_nonparametric_nan_calibration_returns_all_nan():
    result = _nonparametric_z_scores(
        np.array([-1, 0, 1]), np.array([-2, np.nan, 2])
    )
    assert np.isnan(result).all()
