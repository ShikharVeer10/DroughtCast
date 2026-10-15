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
    """KGE 2012 with normalized additive bias, stable for zero-mean SPEI."""
    p, y = np.asarray(preds).ravel(), np.asarray(targets).ravel()
    valid = np.isfinite(p) & np.isfinite(y)
    p, y = p[valid], y[valid]
    if len(y) < 2 or np.std(y) < 1e-12:
        return float("nan")
    # Correlation is undefined for a constant predictor; assign r=0 and retain
    # explicit variability/bias penalties rather than emitting NaN or warnings.
    correlation = 0.0 if np.std(p) < 1e-12 else float(np.corrcoef(p, y)[0, 1])
    if not np.isfinite(correlation):
        correlation = 0.0
    variability = np.std(p) / np.std(y)
    normalized_bias = (np.mean(p) - np.mean(y)) / np.std(y)
    return float(1 - np.sqrt((correlation - 1) ** 2 + (variability - 1) ** 2 + normalized_bias ** 2))


def drought_event_metrics(preds: np.ndarray, targets: np.ndarray, threshold: float = -1.0) -> dict:
    p, y = np.asarray(preds), np.asarray(targets)
    valid = np.isfinite(p) & np.isfinite(y)
    predicted, observed = p[valid] <= threshold, y[valid] <= threshold
    tp = int(np.sum(predicted & observed))
    fp = int(np.sum(predicted & ~observed))
    fn = int(np.sum(~predicted & observed))
    pod = tp / (tp + fn) if tp + fn else float("nan")
    far = fp / (tp + fp) if tp + fp else float("nan")
    csi = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "pod": pod, "far": far, "csi": csi, "f1": f1}
