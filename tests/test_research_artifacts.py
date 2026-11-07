import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from data.build_dataset import _nonparametric_z_scores
from data.protocol import DATASET_PROTOCOL_VERSION
from data.quality import target_quality_report
from utils.artifacts import load_checkpoint, validate_checkpoint_metadata


def test_nonparametric_transform_bounds_duplicate_tail_values():
    values = np.array([-1000.0, 0.0, 1000.0])
    calibration = np.array([1.0, 1.0, 1.0, 2.0, 3.0])
    result = _nonparametric_z_scores(values, calibration)
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) <= 3.5


def test_target_quality_report_flags_implausible_scale():
    dates = pd.date_range("2000-01-01", periods=3, freq="MS")
    frame = pd.DataFrame(
        {
            "date": dates,
            "site": ["a"] * 3,
            "spei_3": [0.0, 1.0, -1.0],
            "spei_6": [0.0, 20.0, -20.0],
            "spei_12": [0.0, 1.0, -1.0],
        }
    )
    report = target_quality_report(frame)
    assert any("spei_6" in warning for warning in report["warnings"])


def test_checkpoint_metadata_mismatch_is_rejected():
    config = {
        "data": {"input_dim": 2, "forecast_steps": 4},
        "model": {"encoder": "lstm", "attention": "shared"},
    }
    checkpoint = {
        "artifact_metadata": {
            "architecture_fingerprint": "wrong",
            "dataset_schema_version": DATASET_PROTOCOL_VERSION,
        }
    }
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint_metadata(checkpoint, config)


def test_checkpoint_without_current_dataset_schema_is_rejected():
    config = {
        "data": {"input_dim": 2, "forecast_steps": 4},
        "model": {"encoder": "lstm", "attention": "shared"},
    }
    checkpoint = {
        "artifact_metadata": {
            "architecture_fingerprint": "compatible",
            "dataset_schema_version": 2,
        }
    }
    with pytest.raises(ValueError, match="dataset schema is incompatible"):
        validate_checkpoint_metadata(checkpoint, config)


def test_tensor_checkpoint_loads_metadata_from_json_sidecar(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"model": {"weight": torch.tensor([1.0])}}, path)
    metadata_path = path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps({
            "config": {"data": {"features": ["x"]}},
            "artifact_metadata": {"dataset_schema_version": DATASET_PROTOCOL_VERSION},
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }),
        encoding="utf-8",
    )
    state = load_checkpoint(path)
    assert state["model"]["weight"].item() == pytest.approx(1.0)
    assert state["artifact_metadata"]["dataset_schema_version"] == DATASET_PROTOCOL_VERSION
    torch.save({"model": {"weight": torch.tensor([2.0])}}, path)
    with pytest.raises(ValueError, match="modified after its metadata"):
        load_checkpoint(path)
