import pytest
import torch

from experiments.run_phase2_5_1b_objective import (
    PRIMARY_MSE_WEIGHT,
    combined_objective,
    primary_mse_loss,
)


def _loss_inputs():
    targets = torch.zeros((2, 3, 4), dtype=torch.float32)
    targets[:, 0, 2:4] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    predictions = {
        name: torch.zeros((2, 4, 3), dtype=torch.float32, requires_grad=True)
        for name in ("spei_3", "spei_6", "spei_12")
    }
    config = {"training": {"extreme_weight": 1.5, "extreme_threshold": -1.0}}
    return predictions, targets, mask, config


def test_alternative_objective_is_pinball_plus_weighted_primary_mse():
    predictions, targets, mask, config = _loss_inputs()
    total, pinball, primary_mse = combined_objective(predictions, targets, mask, config)

    # Pinball sum: SPEI-3 primary target sum=10, 3 quantiles have mean
    # absolute-error multiplier 1.5, divided by 8 valid SPEI-3 targets.
    torch.testing.assert_close(pinball, torch.tensor(1.875))
    torch.testing.assert_close(primary_mse, torch.tensor(7.5))
    torch.testing.assert_close(total, pinball + PRIMARY_MSE_WEIGHT * primary_mse)
    total.backward()
    assert all(torch.isfinite(value.grad).all() for value in predictions.values())


def test_primary_mse_ignores_masked_nonfinite_values():
    prediction = torch.tensor([[1.0, float("inf")]], requires_grad=True)
    target = torch.tensor([[3.0, float("nan")]])
    mask = torch.tensor([[True, False]])

    loss = primary_mse_loss(prediction, target, mask)
    torch.testing.assert_close(loss, torch.tensor(4.0))
    loss.backward()
    assert torch.isfinite(prediction.grad[0, 0])
    assert prediction.grad[0, 1] == 0


def test_primary_mse_rejects_unmasked_nonfinite_values():
    prediction = torch.tensor([[float("nan"), 1.0]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False]])

    with pytest.raises(ValueError, match="finite predictions and targets"):
        primary_mse_loss(prediction, target, mask)


def test_primary_mse_all_masked_returns_zero():
    prediction = torch.tensor([[2.0, 3.0]], requires_grad=True)
    target = torch.tensor([[float("nan"), float("inf")]])
    mask = torch.zeros_like(target, dtype=torch.bool)

    loss = primary_mse_loss(prediction, target, mask)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
