"""Architecture conformance: default model must match the canonical diagram."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from models.config_schema import validate_config
from models.forecaster import DroughtCast
from models.decoder import ForecastDecoder


DIAGRAM = Path("configs/base_diagram.yaml")


@pytest.fixture(scope="module")
def diagram_cfg():
    assert DIAGRAM.is_file(), "configs/base_diagram.yaml is required"
    cfg = yaml.safe_load(DIAGRAM.read_text(encoding="utf-8"))
    validate_config(cfg)
    cfg = dict(cfg)
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["input_dim"] = len(cfg["data"]["features"])
    return cfg


@pytest.fixture(scope="module")
def model(diagram_cfg):
    return DroughtCast(diagram_cfg)


def test_encoder_dims_match_diagram(model):
    assert model.encoder is not None
    assert model.encoder.lstm1.hidden_size == 64
    assert model.encoder.dropout.p == pytest.approx(0.2)
    assert model.encoder.lstm2.hidden_size == 32


def test_encoder_output_shape(model):
    x = torch.randn(4, 6, 12)
    h = model.encoder(x)
    assert h.shape == (4, 6, 32)


def test_attention_weights_shape_sum_and_nonneg(model):
    x = torch.randn(4, 6, 12)
    preds, aux = model(x)
    alpha = aux["attention"]
    assert alpha.shape == (4, 6, 1)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(4, 1), atol=1e-5)
    assert torch.all(alpha >= 0)


def test_shared_context_feeds_all_heads(model):
    x = torch.randn(3, 6, 12)
    _, aux = model(x)
    contexts = aux["contexts"]
    assert len(contexts) == 3
    assert contexts[0].shape == (3, 32)
    assert torch.equal(contexts[0], contexts[1])
    assert torch.equal(contexts[0], contexts[2])
    assert aux["context"].shape == (3, 32)


def test_three_direct_decoders_take_context_only(model):
    assert model.decoders is not None
    assert len(model.decoders) == 3
    for dec in model.decoders:
        assert isinstance(dec, ForecastDecoder)
        first = dec.net[0]
        assert isinstance(first, torch.nn.Linear)
        assert first.in_features == 32


def test_output_predictions_shape(model):
    x = torch.randn(5, 6, 12)
    preds, _ = model(x)
    stacked = torch.stack([preds[k] for k in ("spei_3", "spei_6", "spei_12")], dim=1)
    assert stacked.shape == (5, 3, 4)
    assert model.predict_tensor(x).shape == (5, 3, 4)


def test_param_count_written(model):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DroughtCast base_diagram parameter count: {n_params}")
    out = Path("results/param_count.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": "configs/base_diagram.yaml",
        "parameter_count": int(n_params),
        "architecture": "LSTM(64)->Dropout(0.2)->LSTM(32); additive shared attention; 3 direct heads",
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    stored = json.loads(out.read_text(encoding="utf-8"))
    assert stored["parameter_count"] == n_params
    assert n_params > 0


def test_rejects_dead_config_keys(diagram_cfg):
    bad = yaml.safe_load(DIAGRAM.read_text(encoding="utf-8"))
    bad["model"]["hidden_dim"] = 64
    with pytest.raises(ValueError, match="Dead/removed|Unknown"):
        DroughtCast({**bad, "data": {**bad["data"], "input_dim": 12}})


def test_rejects_unknown_config_keys(diagram_cfg):
    bad = yaml.safe_load(DIAGRAM.read_text(encoding="utf-8"))
    bad["model"]["use_teleconnection_gate"] = True
    with pytest.raises(ValueError):
        DroughtCast({**bad, "data": {**bad["data"], "input_dim": 12}})
