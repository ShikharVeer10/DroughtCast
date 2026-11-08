import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.loader import load_site_table
from data.protocol import MODEL_FEATURES
from models.forecaster import MultiScaleForecaster
from train import pinball_loss

ROOT = Path(__file__).resolve().parents[1]


def test_pinball_manual_positive_negative_zero_and_masked_target():
    prediction = torch.zeros((1, 4, 3), dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[2.0, -2.0, 0.0, 999.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True, True, False]])
    actual = pinball_loss(prediction, target, mask, [0.1, 0.5, 0.9])
    # The positive and negative examples each sum to 3 across q10/q50/q90;
    # zero error contributes 0 and the masked fourth target contributes 0.
    torch.testing.assert_close(actual, torch.tensor(2.0, dtype=torch.float64))
    actual.backward()
    assert torch.count_nonzero(prediction.grad[:, 3]) == 0
    assert torch.isfinite(prediction.grad).all()


def test_pinball_manual_quantile_error_direction_and_zero():
    q = [0.1, 0.5, 0.9]
    pred = torch.tensor([[[0.0, 0.0, 0.0]]])
    mask = torch.ones((1, 1), dtype=torch.bool)
    positive = pinball_loss(pred, torch.tensor([[1.0]]), mask, q)
    negative = pinball_loss(pred, torch.tensor([[-1.0]]), mask, q)
    zero = pinball_loss(pred, torch.tensor([[0.0]]), mask, q)
    torch.testing.assert_close(positive, torch.tensor(1.5))
    torch.testing.assert_close(negative, torch.tensor(1.5))
    torch.testing.assert_close(zero, torch.tensor(0.0))


def test_masked_infinity_has_zero_contribution_and_finite_gradients():
    prediction = torch.zeros((1, 2, 3), dtype=torch.float32)
    target = torch.tensor([[0.0, float("inf")]], dtype=torch.float32)
    prediction.requires_grad_()
    mask = torch.tensor([[True, False]])
    loss = pinball_loss(prediction, target, mask, [0.1, 0.5, 0.9])
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_model_head_scale_lead_quantile_mapping_and_gradient_flow():
    metadata = json.loads(
        (ROOT / "results/maharashtra/phase2/neural/"
         "protocol10_seed42_e9dfa7ce76_ae290b4115_1d7a5ce0/"
         "neural_experiment_metadata.json").read_text(encoding="utf-8")
    )
    model = MultiScaleForecaster(metadata["selected_config"])
    model.eval()
    x = torch.randn(3, 6, 15)
    output, _ = model(x)
    scale_order = ("spei_3", "spei_6", "spei_12")
    assert tuple(output) == scale_order
    assert all(output[name].shape == (3, 4, 3) for name in scale_order)
    stacked_median = model.predict_tensor(x)
    assert stacked_median.shape == (3, 3, 4)
    torch.testing.assert_close(
        stacked_median,
        torch.stack([output[name][..., 1] for name in scale_order], dim=1),
    )
    for name in scale_order:
        assert torch.all(output[name][..., 0] <= output[name][..., 1])
        assert torch.all(output[name][..., 1] <= output[name][..., 2])

    targets = torch.randn(3, 3, 4)
    mask = torch.ones_like(targets, dtype=torch.bool)
    loss = sum(
        pinball_loss(output[name], targets[:, index], mask[:, index], [0.1, 0.5, 0.9])
        for index, name in enumerate(scale_order)
    )
    loss.backward()
    groups = {
        "lstm1": [p for n, p in model.named_parameters() if "lstm1" in n],
        "lstm2": [p for n, p in model.named_parameters() if "lstm2" in n],
        "attention": [p for n, p in model.named_parameters() if "attentions" in n],
        "spei3_head": list(model.decoders[0].parameters()),
        "spei6_head": list(model.decoders[1].parameters()),
        "spei12_head": list(model.decoders[2].parameters()),
    }
    for parameters in groups.values():
        gradients = [p.grad for p in parameters]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_loader_keeps_targets_in_native_spei_units(tmp_path):
    dates = pd.date_range("1990-01-01", "1990-12-01", freq="MS")
    data = {"date": dates, "site": "akola"}
    for index, feature in enumerate(MODEL_FEATURES):
        data[feature] = np.arange(len(dates), dtype=float) + index
    for name in ("GWETTOP", "GWETROOT"):
        data[name] = np.full(len(dates), 0.5)
    for scale_index, name in enumerate(("spei_3", "spei_6", "spei_12")):
        data[name] = scale_index + np.arange(len(dates), dtype=float) / 10
    path = tmp_path / "raw_spei_targets.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    dataset = load_site_table(
        path, features=list(MODEL_FEATURES), seq_len=6, forecast_steps=4,
        site_filter=["akola"], include_splits=("train",),
    )["train"]
    expected = np.array([
        np.arange(6, 10) / 10,
        1 + np.arange(6, 10) / 10,
        2 + np.arange(6, 10) / 10,
    ], dtype=np.float32)
    np.testing.assert_allclose(dataset.Y[0].numpy(), expected)
