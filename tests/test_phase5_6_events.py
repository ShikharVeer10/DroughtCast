import numpy as np
import pandas as pd

from experiments.phase5_6_events import categories, classification_metrics


def test_category_boundaries_are_right_closed():
    assert categories(np.array([-2.0, -1.5, -1.0, 0.0, 1.0, 1.01])).tolist() == [0, 1, 2, 3, 3, 4]


def test_confusion_metrics_and_undefined_values():
    frame = pd.DataFrame({"observed": [-2.0, -1.5, -1.0, 0.0], "prediction": [-2.0, -1.0, -1.0, 0.0]})
    matrix, metrics = classification_metrics(frame)
    assert matrix.sum() == 4
    assert metrics[0]["severe_recall"] == 0.0
    _, undefined = classification_metrics(pd.DataFrame({"observed": [0.0], "prediction": [0.0]}))
    assert np.isnan(undefined[0]["severe_precision"])
