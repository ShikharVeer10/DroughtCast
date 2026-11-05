import numpy as np

from data.build_dataset import _nonparametric_z_scores, _normalize_sites, SITES


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


def test_normalize_sites_dict():
    sample = {"site1": {"lat": 10.0, "lon": 20.0, "region": "r1"}}
    assert _normalize_sites(sample) == sample


def test_normalize_sites_list_of_dicts():
    raw_list = [
        {"id": "aurangabad", "name": "Aurangabad", "region": "Marathwada", "lat": 19.8762, "lon": 75.3433},
        {"id": "latur", "name": "Latur", "region": "Marathwada", "lat": 18.4088, "lon": 76.5604},
    ]
    normalized = _normalize_sites(raw_list)
    assert "aurangabad" in normalized
    assert "latur" in normalized
    assert normalized["aurangabad"]["lat"] == 19.8762
    assert normalized["aurangabad"]["region"] == "Marathwada"


def test_normalize_sites_empty_or_none():
    assert _normalize_sites(None) == SITES
    assert _normalize_sites([]) == SITES
