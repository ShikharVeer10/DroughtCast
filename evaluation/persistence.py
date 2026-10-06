import numpy as np

class PersistenceBaseline:
    def __init__(self, target_feature_idx: int = 0):
        self.target_feature_idx = target_feature_idx

    def predict(self, x_inputs: np.ndarray, forecast_steps: int = 4) -> np.ndarray:
        last_observed = x_inputs[:, -1, self.target_feature_idx]
        predictions = np.repeat(last_observed[:, np.newaxis], forecast_steps, axis=1)
        return predictions


class ClimatologyBaseline:
    def __init__(self, historical_mean: float = 0.0):
        self.historical_mean = historical_mean

    def predict(self, num_samples: int, forecast_steps: int = 4) -> np.ndarray:
        return np.full((num_samples, forecast_steps), self.historical_mean)