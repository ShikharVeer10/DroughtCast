import numpy as np

def calculate_rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - targets) ** 2)))

def calculate_mae(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - targets)))

def calculate_nse(preds: np.ndarray, targets: np.ndarray) -> float:
    numerator = np.sum((targets - preds) ** 2)
    denominator = np.sum((targets - np.mean(targets)) ** 2)
    return float(1.0 - (numerator / (denominator + 1e-8)))

def calculate_kge(preds: np.ndarray, targets: np.ndarray) -> float:
    r = np.corrcoef(preds.flatten(), targets.flatten())[0, 1]
    alpha = np.std(preds) / (np.std(targets) + 1e-8)
    beta = np.mean(preds) / (np.mean(targets) + 1e-8)
    kge = 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return float(kge)