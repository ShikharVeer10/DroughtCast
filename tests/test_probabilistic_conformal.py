import numpy as np
import pytest
import torch

from models.forecaster import MultiScaleForecaster
from utils.conformal import (
    apply_cqr,
    calibrate_cqr,
    crps_from_quantiles,
    interval_metrics,
)
from utils.interpretability import integrated_gradients


def test_quantile_outputs_are_ordered():
    cfg = {
        "data": {"input_dim": 4, "forecast_steps": 2},
        "model": {
            "encoder": "lstm",
            "attention": "shared",
            "quantiles": [0.1, 0.5, 0.9],
            "hidden_dim_1": 8,
            "hidden_dim_2": 8,
        },
    }
    output, _ = MultiScaleForecaster(cfg)(torch.randn(3, 6, 4))
    for prediction in output.values():
        assert prediction.shape == (3, 2, 3)
        assert torch.all(prediction[..., 0] <= prediction[..., 1])
        assert torch.all(prediction[..., 1] <= prediction[..., 2])


def test_cqr_uses_validation_scores_and_reports_coverage():
    val_y = np.array([[0.0], [1.0], [-1.0], [2.0]])
    val_pred = np.zeros((4, 1, 3))
    mask = np.ones((4, 1), dtype=bool)
    correction = calibrate_cqr(val_pred, val_y, mask, [0.1, 0.5, 0.9], coverage=0.8)
    assert correction.shape == (1,)
    test_y = np.array([[-2.0], [0.0], [2.0]])
    test_pred = np.zeros((3, 1, 3))
    lower, upper = apply_cqr(test_pred, correction)
    metrics = interval_metrics(lower[:, 0], upper[:, 0], test_y[:, 0])
    assert metrics["n"] == 3
    assert 0 <= metrics["coverage"] <= 1
    assert crps_from_quantiles(test_pred, test_y, [0.1, 0.5, 0.9]) >= 0


def test_cqr_preserves_negative_correction_without_silent_clipping():
    from utils.conformal import apply_cqr

    prediction = np.array([[[0.0, 0.5, 1.0]]])
    lower, upper = apply_cqr(prediction, np.array([-0.25]))
    assert lower[0, 0] == pytest.approx(0.25)
    assert upper[0, 0] == pytest.approx(0.75)


def test_integrated_gradients_has_input_shape_and_finite_values():
    cfg = {
        "data": {"input_dim": 2, "forecast_steps": 2},
        "model": {
            "encoder": "gru",
            "attention": "shared",
            "hidden_dim_1": 4,
            "hidden_dim_2": 4,
            "dropout": 0.0,
        },
    }
    model = MultiScaleForecaster(cfg)
    x = torch.randn(2, 3, 2)
    result = integrated_gradients(model, x, "spei_3", 0, steps=4)
    assert result.shape == x.shape
    assert torch.isfinite(result).all()
