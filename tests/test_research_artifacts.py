import numpy as np
import pandas as pd
import pytest

from data.build_dataset import _nonparametric_z_scores
from data.protocol import DATASET_PROTOCOL_VERSION
from data.quality import target_quality_report
from utils.artifacts import validate_checkpoint_metadata


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
