import numpy as np
import pandas as pd
import pytest

from evaluation.robustness import (
    forecast_behavior_summary,
    model_disagreement,
    stability_summary,
)


def test_behavior_and_disagreement_are_reproducible():
    frame = pd.DataFrame([
        {"model": "a", "site": "x", "scale": "spei_3", "lead": "T+1", "prediction": 1.0},
        {"model": "b", "site": "x", "scale": "spei_3", "lead": "T+1", "prediction": 3.0},
    ])
    summary = forecast_behavior_summary(frame)
    disagreement = model_disagreement(frame)
    assert len(summary) == 2
    assert disagreement.loc[0, "model_range"] == 2.0


def test_stability_empty_values_remain_nan():
    result = stability_summary(pd.DataFrame({"metric": [np.nan]}))
    assert result["n"] == 0
    assert np.isnan(result["mean"])
    with pytest.raises(ValueError):
        stability_summary(pd.DataFrame({"other": [1]}))
