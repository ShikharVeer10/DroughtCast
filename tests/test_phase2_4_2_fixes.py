import json
from pathlib import Path

import pytest
import torch
import yaml

from experiments.run_phase2_2_neural import _model_config
from models.forecaster import MultiScaleForecaster
from train import pinball_loss

ROOT = Path(__file__).resolve().parents[1]
QUANTILES = (0.1, 0.5, 0.9)


@pytest.mark.parametrize("masked_target", [float("nan"), float("inf"), -float("inf")])
def test_masked_nonfinite_targets_do_not_enter_loss_or_gradients(masked_target):
    prediction = torch.zeros((1, 2, 3), dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[1.0, masked_target]], dtype=torch.float64)
    mask = torch.tensor([[True, False]])

    loss = pinball_loss(prediction, target, mask, QUANTILES)
    # Existing reduction sums quantile losses and divides by valid target cells.
    torch.testing.assert_close(loss, torch.tensor(1.5, dtype=torch.float64))
    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[:, 1]) == 0


@pytest.mark.parametrize("invalid_target", [float("nan"), float("inf"), -float("inf")])
def test_unmasked_nonfinite_targets_are_rejected(invalid_target):
    prediction = torch.zeros((1, 1, 3))
    target = torch.tensor([[invalid_target]])
    mask = torch.ones((1, 1), dtype=torch.bool)

    with pytest.raises(ValueError, match="Unmasked targets must be finite"):
        pinball_loss(prediction, target, mask, QUANTILES)


def test_all_masked_targets_return_zero_with_finite_zero_gradients():
    prediction = torch.randn((2, 4, 3), requires_grad=True)
    target = torch.full((2, 4), float("nan"))
    mask = torch.zeros((2, 4), dtype=torch.bool)

    loss = pinball_loss(prediction, target, mask, QUANTILES)
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) == 0


def test_model_config_serializes_the_effective_disabled_gate():
    base = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    profile = {
        "hidden_dim_1": 64,
        "hidden_dim_2": 32,
        "dropout": 0.3,
        "learning_rate": 0.0005,
        "weight_decay": 0.0005,
        "batch_size": 64,
    }
    resolved = _model_config(base, profile)
    serialized = json.loads(json.dumps(resolved))

    assert serialized["model"]["variable_gate"] is False
    assert "use_variable_gate" not in serialized["model"]
    model = MultiScaleForecaster(serialized)
    assert model.variable_gate is None
    # Resolving an experiment config must not mutate the source configuration.
    assert "use_variable_gate" not in base["model"]


def test_historical_checkpoint_configuration_keeps_gate_disabled():
    metadata_path = (
        ROOT
        / "results/maharashtra/phase2/neural/"
        / "protocol10_seed42_e9dfa7ce76_ae290b4115_1d7a5ce0/"
        / "neural_experiment_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    historical_config = json.loads(json.dumps(metadata["selected_config"]))
    model = MultiScaleForecaster(historical_config)

    assert historical_config["model"]["variable_gate"] is False
    assert model.variable_gate is None
