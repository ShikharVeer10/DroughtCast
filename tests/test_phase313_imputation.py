import json
from pathlib import Path

import numpy as np

from data.preprocessing import ClimateNormalizer


def test_frozen_training_mean_imputation_normalizes_missing_values_to_zero():
    metadata = json.loads(
        Path("data/normalizers/maharashtra_protocol10.json").read_text()
    )
    site = metadata["normalizers"]["akola"]
    normalizer = ClimateNormalizer.from_dict(site, metadata["features"])
    values = np.full((1, len(metadata["features"])), np.nan)
    transformed = normalizer.transform(values)
    assert np.allclose(transformed, 0.0)
