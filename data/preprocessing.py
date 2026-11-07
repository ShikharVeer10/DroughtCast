import numpy as np


class ClimateNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("Expected [time, feature] data")
        finite = np.isfinite(data)
        counts = finite.sum(axis=0)
        if np.any(counts == 0):
            raise ValueError("Every feature needs at least one finite training observation")
        values = np.where(finite, data, np.nan)
        self.mean = np.nanmean(values, axis=0)
        self.std = np.nanstd(values, axis=0)
        self.std = np.where(np.isfinite(self.std) & (self.std > 0), self.std, 1.0)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ValueError("ClimateNormalizer must be fitted before transform")
        values = np.asarray(data, dtype=float)
        if values.shape[-1] != len(self.mean):
            raise ValueError("Feature count does not match fitted normalizer")
        imputed = np.where(np.isfinite(values), values, self.mean)
        return (imputed - self.mean) / self.std

    def to_dict(self):
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "imputation": "training_feature_mean",
        }

    @classmethod
    def from_dict(cls, metadata: dict) -> "ClimateNormalizer":
        normalizer = cls()
        normalizer.mean = np.asarray(metadata.get("mean"), dtype=float)
        normalizer.std = np.asarray(metadata.get("std"), dtype=float)
        if (
            normalizer.mean.ndim != 1
            or normalizer.std.shape != normalizer.mean.shape
            or not np.isfinite(normalizer.mean).all()
            or not np.isfinite(normalizer.std).all()
            or np.any(normalizer.std <= 0)
        ):
            raise ValueError("Invalid saved ClimateNormalizer statistics")
        return normalizer

