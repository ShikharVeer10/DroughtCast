import json

import numpy as np
import pandas as pd
import pytest

from data.build_dataset import _nonparametric_z_scores
from data.quality import target_quality_report
from utils.artifacts import architecture_fingerprint, validate_checkpoint_metadata


def test_nonparametric_transform_bounds_duplicate_tail_values():
    values = np.array([-1000.0, 0.0, 1000.0])
    calibration = np.array([1.0, 1.0, 1.0, 2.0, 3.0])
    result = _nonparametric_z_scores(values, calibration)
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) <= 8.0


def test_target_quality_report_flags_implausible_scale():
    dates = pd.date_range("2000-01-01", periods=3, freq="MS")
    frame = pd.DataFrame({
        "date": dates, "site": ["a"] * 3,
        "spei_3": [0.0, 1.0, -1.0],
        "spei_6": [0.0, 20.0, -20.0],
        "spei_12": [0.0, 1.0, -1.0],
    })
    report = target_quality_report(frame)
    assert any("spei_6" in warning for warning in report["warnings"])


def test_checkpoint_metadata_mismatch_is_rejected():
    config = {"data": {"input_dim": 2, "forecast_steps": 4},
              "model": {"encoder": "lstm", "attention": "shared"}}
    checkpoint = {"artifact_metadata": {"architecture_fingerprint": "wrong"}}
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint_metadata(checkpoint, config)
