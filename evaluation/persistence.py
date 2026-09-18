import numpy as np

class PersistenceBaseline:
    def predict(self, x_inputs: np.ndarray, forecast_steps: int = 4) -> np.ndarray:
        last_observed = x_inputs[:, -1, 0:1]
        return np.repeat(last_observed, forecast_steps, axis=1)

class ClimatologyBaseline:
    def __init__(self, historical_mean: float):
        self.historical_mean = historical_mean

    def predict(self, num_samples: int, forecast_steps: int = 4) -> np.ndarray:
        return np.full((num_samples, forecast_steps), self.historical_mean)