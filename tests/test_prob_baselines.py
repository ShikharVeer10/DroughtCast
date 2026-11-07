import numpy as np
import pytest

from utils.prob_baselines import persistence_residual_quantiles


def test_persistence_residual_quantiles_are_train_only_and_ordered():
    train_last = np.zeros((40, 3))
    train_targets = np.zeros((40, 3, 2))
    residuals = np.linspace(-2.0, 2.0, 40)
    train_targets[:, :, 0] = residuals[:, None]
    train_targets[:, :, 1] = (2 * residuals)[:, None]
    mask = np.ones_like(train_targets, dtype=bool)
    pred = persistence_residual_quantiles(
        np.array([[1.0, 2.0, 3.0]]), train_last, train_targets, mask, [0.1, 0.5, 0.9]
    )
    assert pred.shape == (1, 3, 2, 3)
    assert np.all(np.diff(pred, axis=-1) > 0)
    assert pred[0, 0, 0, 1] == pytest.approx(1.0)


def test_persistence_residual_quantiles_rejects_small_training_sample():
    with pytest.raises(ValueError, match="Too few training residuals"):
        persistence_residual_quantiles(
            np.zeros((2, 1)), np.zeros((9, 1)), np.zeros((9, 1, 1)),
            np.ones((9, 1, 1), dtype=bool), [0.1, 0.5, 0.9]
        )
