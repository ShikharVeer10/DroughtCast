from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from data.loader import load_site_table
from data.preprocessing import validate_normalizer_metadata
from data.protocol import DATASET_PROTOCOL_VERSION, MODEL_FEATURES, SPLIT_BOUNDARIES
from experiments.run_phase2_2_neural import (
    NORMALIZER_PATH,
    QUANTILES,
    TARGETS,
    ValidationEarlyStopper,
    _model_config,
    _quantile_predictions,
    drought_categories,
    seed_everything,
)
from models.forecaster import MultiScaleForecaster
from train import pinball_loss


def _config():
    return {
        "data": {"seq_len": 6, "input_dim": 15, "forecast_steps": 4, "features": MODEL_FEATURES},
        "model": {
            "encoder": "lstm",
            "attention": "shared",
            "heads": "direct",
            "input_dim": 15,
            "hidden_dim_1": 8,
            "hidden_dim_2": 4,
            "dropout": 0.0,
            "quantiles": list(QUANTILES),
            "use_cascaded_decoder": False,
            "variable_gate": False,
        },
        "training": {"learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 2},
    }


def test_model_output_shape_scale_lead_and_quantile_mapping():
    seed_everything(42)
    model = MultiScaleForecaster(_config()).eval()
    x = torch.randn(3, 6, 15)
    predictions, _ = model(x)
    assert tuple(predictions) == TARGETS
    for scale in TARGETS:
        values = predictions[scale]
        assert values.shape == (3, 4, 3)
        assert torch.isfinite(values).all()
        assert torch.all(values[..., 0] <= values[..., 1])
        assert torch.all(values[..., 1] <= values[..., 2])
    # This verifies the production conversion returns scale x lead medians.
    median_output = model.predict_tensor(x)
    assert median_output.shape == (3, 3, 4)
    for scale_index, scale in enumerate(TARGETS):
        torch.testing.assert_close(median_output[:, scale_index], predictions[scale][..., 1])


def test_model_config_omits_dataset_builder_only_setting_without_mutating_source():
    base = _config()
    base["data"]["spei_calibration_start_year"] = 1990
    profile = {
        "hidden_dim_1": 8,
        "hidden_dim_2": 4,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 2,
    }
    resolved = _model_config(base, profile)
    assert "spei_calibration_start_year" not in resolved["data"]
    assert base["data"]["spei_calibration_start_year"] == 1990
    assert isinstance(MultiScaleForecaster(resolved), MultiScaleForecaster)


def test_pinball_loss_masks_targets_and_backpropagates():
    prediction = torch.tensor([[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]], requires_grad=True)
    target = torch.tensor([[1.0, 99.0]])
    mask = torch.tensor([[True, False]])
    loss = pinball_loss(prediction, target, mask, QUANTILES)
    assert loss.item() == pytest.approx(0.2)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[:, 1]) == 0


def test_validation_prediction_has_no_gradients_or_parameter_updates():
    model = MultiScaleForecaster(_config())
    before = [parameter.detach().clone() for parameter in model.parameters()]
    values = _quantile_predictions(model, torch.ones(2, 6, 15), torch.device("cpu"))
    assert all(values[name].shape == (2, 4, 3) for name in TARGETS)
    assert all(parameter.grad is None for parameter in model.parameters())
    for parameter, original in zip(model.parameters(), before):
        torch.testing.assert_close(parameter, original)


def test_validation_early_stopper_tracks_best_and_patience():
    stopper = ValidationEarlyStopper(patience=2, min_delta=0.01)
    assert stopper.update(0.1, 1)
    assert not stopper.update(0.105, 2)
    assert not stopper.should_stop
    assert stopper.update(0.2, 3)
    assert not stopper.update(0.19, 4)
    assert not stopper.should_stop
    assert not stopper.update(0.19, 5)
    assert stopper.should_stop
    assert stopper.best_epoch == 3
    with pytest.raises(ValueError, match="must be finite"):
        stopper.update(float("nan"), 6)


def test_drought_category_boundaries_and_missing_values():
    values = np.array([-3.0, -2.0, -1.999, -1.5, -1.499, -1.0, -0.999, 0.999, 1.0, 1.5, 2.0, np.nan])
    np.testing.assert_array_equal(
        drought_categories(values),
        np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, -1], dtype=np.int8),
    )


def test_verified_normalizer_has_protocol10_order_and_finite_stats():
    import json

    metadata = json.loads(NORMALIZER_PATH.read_text(encoding="utf-8"))
    validate_normalizer_metadata(
        metadata,
        list(MODEL_FEATURES),
        expected_protocol_version=DATASET_PROTOCOL_VERSION,
        expected_split_boundaries=SPLIT_BOUNDARIES,
        expected_seq_len=6,
        expected_forecast_steps=4,
    )
    assert metadata["features"] == list(MODEL_FEATURES)
    assert metadata["training_split"] == "train"
    assert all(len(stats["mean"]) == len(stats["std"]) == 15 for stats in metadata["normalizers"].values())


def test_checkpoint_weights_only_round_trip(tmp_path):
    seed_everything(2024)
    model = MultiScaleForecaster(_config())
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = MultiScaleForecaster(_config())
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    for left, right in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right)


def test_seeded_initialization_is_reproducible():
    seed_everything(101)
    first = MultiScaleForecaster(_config())
    seed_everything(101)
    second = MultiScaleForecaster(_config())
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right)


def test_training_step_updates_model_parameters():
    model = MultiScaleForecaster(_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4, 6, 15)
    target = torch.randn(4, 3, 4)
    mask = torch.ones_like(target, dtype=torch.bool)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    predictions, _ = model(x)
    loss = sum(
        pinball_loss(predictions[name], target[:, index], mask[:, index], QUANTILES)
        for index, name in enumerate(TARGETS)
    )
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_validation_evaluator_keeps_site_and_scale_dimensions_visible():
    from experiments.run_phase2_2_neural import _evaluate_predictions

    y = np.zeros((2, 3, 4), dtype=np.float32)
    y[0, 0, 2] = -1.7
    y[1, 0, 2] = -2.1
    dataset = SimpleNamespace(
        Y=torch.from_numpy(y),
        mask=torch.ones((2, 3, 4), dtype=torch.bool),
        metadata=[{"site": "akola"}, {"site": "beed"}],
    )
    predictions = {
        name: np.zeros((2, 4, 3), dtype=np.float32)
        for name in TARGETS
    }
    metrics, confusion, _ = _evaluate_predictions(
        dataset,
        predictions,
        split="validation",
        seed=42,
        run_tag="test",
        sites=["akola", "beed"],
    )
    assert set(metrics["site"]) == {"akola", "beed", "ALL"}
    row = metrics.loc[(metrics.target_column == "spei_3") & (metrics.lead == "T+3") & (metrics.site == "ALL")].iloc[0]
    assert row["n_valid"] == 2
    assert row["severe_or_extreme_threshold"] == "SPEI <= -1.5"
    assert row["severe_drought_threshold"] == "-2.0 < SPEI <= -1.5"
    assert confusion["validation|test|spei_3|T+3|ALL"]["n_valid"] == 2


def test_loader_can_build_train_validation_without_materializing_test(tmp_path):
    dates = pd.date_range("1990-01-01", "2025-12-01", freq="MS")
    data = {"date": dates, "site": "akola"}
    for index, feature in enumerate(MODEL_FEATURES):
        data[feature] = np.arange(len(dates), dtype=float) + index
    for target_index, target in enumerate(TARGETS):
        data[target] = np.arange(len(dates), dtype=float) / 100 + target_index
    frame = pd.DataFrame(data)
    test_rows = frame["date"] >= pd.Timestamp(SPLIT_BOUNDARIES["test_start"])
    frame.loc[test_rows, list(MODEL_FEATURES)] = 1e20
    frame.loc[test_rows, list(TARGETS)] = 1e20
    path = tmp_path / "monthly.csv"
    frame.to_csv(path, index=False)

    datasets = load_site_table(
        path,
        features=list(MODEL_FEATURES),
        seq_len=6,
        forecast_steps=4,
        split_config=SPLIT_BOUNDARIES,
        include_splits=("train", "val"),
    )
    assert len(datasets["train"]) == 243
    assert len(datasets["val"]) == 39
    assert len(datasets["test"]) == 0
    assert max(item["target_dates"][-1] for item in datasets["val"].metadata) <= "2015-12-01"
    assert torch.isfinite(datasets["val"].X).all()
