import numpy as np
import pandas as pd
import pytest

from evaluation.error_analysis import (
    drought_failure_summary,
    require_observed_evaluation,
    residual_summary,
)


def test_residual_and_event_failure_summaries():
    frame = pd.DataFrame([
        {"model": "m", "site": "a", "scale": "spei_3", "lead": "T+1",
         "prediction": -1.0, "observed": -2.0},
        {"model": "m", "site": "a", "scale": "spei_3", "lead": "T+1",
         "prediction": -2.0, "observed": -1.0},
    ])
    result = residual_summary(frame)
    events = drought_failure_summary(frame)
    assert result.loc[0, "n"] == 2
    assert events["missed_events"] == 1


def test_missing_observations_are_rejected():
    frame = pd.DataFrame({"prediction": [0.0], "observed": [np.nan]})
    with pytest.raises(ValueError, match="genuine observed"):
        require_observed_evaluation(frame)
