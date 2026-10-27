"""Train-partition-only, NaN-aware feature standardisation."""

import numpy as np


class ClimateNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("Expected [time, feature] data")
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)
        self.std = np.where(np.isfinite(self.std) & (self.std > 0), self.std, 1.0)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ValueError("ClimateNormalizer must be fitted before transform")
        return (np.asarray(data, dtype=float) - self.mean) / self.std

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}
