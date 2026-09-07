import numpy as np


class ClimateNormalizer:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        self.mean = np.mean(data, axis=(0, 1))
        self.std = np.std(data, axis=(0, 1))

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("ClimateNormalizer must be fitted before transform.")
        return (data - self.mean) / (self.std + 1e-8)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)