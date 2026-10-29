import numpy as np
import pytest

from experiments.build_probability_ensemble import aggregate_seed_probabilities
from experiments.export_probability_table import format_probability, format_scale_label
from utils.probability import (
    event_z,
    fit_platt_calibrator,
    quantile_risk_below_threshold,
    raw_event_probability,
)


def test_quantile_nodes_and_distinguishes_tail_forecasts():
    q = np.array([[-1.2815515655, 0.0, 1.2815515655]])
    assert np.allclose(raw_event_probability(event_z(q, -1.2815515655)), 0.1)
    assert np.allclose(raw_event_probability(event_z(q, 0)), 0.5)
    assert np.allclose(raw_event_probability(event_z(q, 1.2815515655)), 0.9)
    first, second = np.array([1.5, 2, 2.5]), np.array([-0.6, 0.2, 0.9])
    assert quantile_risk_below_threshold(first, -1) == quantile_risk_below_threshold(second, -1)
    assert event_z(first, -1) < event_z(second, -1)


def test_event_z_validation_and_collapsed_interval():
    with pytest.raises(ValueError):
        event_z([1, 0, 2], 0)
    assert np.isfinite(event_z([0, 0, 0], -1))


def test_platt_synthetic_resolution_and_skill():
    rng = np.random.default_rng(0)
    mu = rng.normal(size=3300)
    sd = rng.uniform(0.4, 1.2, size=3300)
    event = (mu + sd * rng.normal(size=3300) <= -1).astype(int)
    z = (-1 - mu) / sd
    calibrator = fit_platt_calibrator(z[:300], event[:300])
    probability = calibrator.predict(z[300:])
    base = event[300:].mean()
    assert np.isfinite(probability).all()
    assert np.all((probability>=0.0) & (probability<=1.0))
    assert np.unique(probability[(probability > 0.005) & (probability < 0.995)]).size > 1
    assert np.mean((probability - event[300:]) ** 2) < np.mean((base - event[300:]) ** 2)


def test_platt_class_validation_and_low_event_warning():
    with pytest.raises(ValueError):
        fit_platt_calibrator([0, 1], [1, 1])
    with pytest.warns(UserWarning, match="Only 3 events"):
        fit_platt_calibrator(np.arange(10), [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])


def test_probability_table_formatter_uses_cutoff_and_scale_labels():
    assert format_probability(0.0075) == "0.0075"
    assert format_probability(0.9908) == "0.9908"
    assert format_probability(0.00002571166980373012) == "0.00002571166980373012"
    assert format_probability(0.9999) == "0.9999"
    with pytest.raises(ValueError):
        format_probability(1.0)
    assert format_scale_label("SPEI_3") == "SPEI-3"


def test_ensemble_mean_is_between_seed_min_and_max():
    import pandas as pd

    base = {
        "site": "phoenix",
        "input_start_month": "2024-07",
        "input_end_month": "2024-12",
        "target_month": "2025-01",
        "scale": "SPEI_3",
        "lead": "T+1",
        "drought_threshold_spei": -1.0,
    }
    result = aggregate_seed_probabilities([
        pd.DataFrame([{**base, "calibrated_probability": value}])
        for value in (0.2, 0.8, 0.5)
    ])
    row = result.iloc[0]
    assert row.calibrated_probability_min <= row.calibrated_probability_mean <= row.calibrated_probability_max


def test_unclipped_platt_prediction_preserve_test_resolution():
    from utils.probability import PlattCalibrator

    calibrator=PlattCalibrator(intercept=-2.0,slope=1.0,n_samples=100,n_events=5)
    z=np.array([-10.0,-9.0,-8.0])
    probability=calibrator.predict(z)
    assert np.isfinite(probability).all()
    assert np.all((probability>=0.0) & (probability<=1.0))
    assert np.unique(probability).size==3
    assert np.all(probability<calibrator.floor)
