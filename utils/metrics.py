import numpy as np


def calculate_rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def calculate_mae(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - targets)))


def calculate_mse(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((preds - targets) ** 2))


def calculate_r_squared(preds: np.ndarray, targets: np.ndarray) -> float:
    p, y = np.asarray(preds).ravel(), np.asarray(targets).ravel()
    valid = np.isfinite(p) & np.isfinite(y)
    p, y = p[valid], y[valid]
    if len(y) < 2 or np.std(y) < 1e-12:
        return float("nan")
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - (ss_res / (ss_tot + 1e-12)))


def skill_score_persistence(
    preds: np.ndarray, targets: np.ndarray, persistence_preds: np.ndarray
) -> float:
    """
    Standard hydrometeorological Skill Score over Persistence (SS_pers):
    SS_pers = 1.0 - (MSE_model / MSE_persistence).
    Positive indicates improvement over persistence benchmark.
    """
    p, y, b = (
        np.asarray(preds).ravel(),
        np.asarray(targets).ravel(),
        np.asarray(persistence_preds).ravel(),
    )
    valid = np.isfinite(p) & np.isfinite(y) & np.isfinite(b)
    p, y, b = p[valid], y[valid], b[valid]
    if len(y) == 0:
        return float("nan")
    mse_model = np.mean((p - y) ** 2)
    mse_pers = np.mean((b - y) ** 2)
    if mse_pers < 1e-12:
        return float("nan")
    return float(1.0 - (mse_model / mse_pers))


def calculate_nse(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Nash-Sutcliffe Efficiency (NSE):
    NSE = 1 - (sum(obs - pred)^2 / sum(obs - mean(obs))^2)
    """
    p, y = np.asarray(preds).ravel(), np.asarray(targets).ravel()
    valid = np.isfinite(p) & np.isfinite(y)
    p, y = p[valid], y[valid]
    if len(y) < 2 or np.std(y) < 1e-12:
        return float("nan")
    numerator = np.sum((y - p) ** 2)
    denominator = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - (numerator / (denominator + 1e-8)))


def calculate_kge(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Kling-Gupta Efficiency (KGE-2012) with normalized additive bias,
    stable for standard normal zero-mean indices such as SPEI:
    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2),
    where alpha = sigma_pred / sigma_obs, beta = (mu_pred - mu_obs) / sigma_obs.
    """
    p, y = np.asarray(preds).ravel(), np.asarray(targets).ravel()
    valid = np.isfinite(p) & np.isfinite(y)
    p, y = p[valid], y[valid]
    if len(y) < 2 or np.std(y) < 1e-12:
        return float("nan")
    correlation = 0.0 if np.std(p) < 1e-12 else float(np.corrcoef(p, y)[0, 1])
    if not np.isfinite(correlation):
        correlation = 0.0
    variability = np.std(p) / np.std(y)
    normalized_bias = (np.mean(p) - np.mean(y)) / np.std(y)
    return float(
        1.0
        - np.sqrt(
            (correlation - 1.0) ** 2 + (variability - 1.0) ** 2 + normalized_bias**2
        )
    )


def drought_event_metrics(
    preds: np.ndarray, targets: np.ndarray, threshold: float = -1.0
) -> dict[str, int | float]:
    """
    Drought onset & extreme regime contingency metrics for threshold:
    - Default threshold = -1.0 (Moderate drought)
    - Severe drought threshold = -1.5
    - Extreme drought threshold = -2.0

    CSI: Critical Success Index (Threat Score) = TP / (TP + FP + FN)
    POD: Probability of Detection (Hit Rate) = TP / (TP + FN)
    FAR: False Alarm Ratio = FP / (TP + FP)
    """
    p, y = np.asarray(preds).ravel(), np.asarray(targets).ravel()
    valid = np.isfinite(p) & np.isfinite(y)
    predicted, observed = p[valid] <= threshold, y[valid] <= threshold
    tp = int(np.sum(predicted & observed))
    fp = int(np.sum(predicted & ~observed))
    fn = int(np.sum(~predicted & observed))
    tn = int(np.sum(~predicted & ~observed))

    pod = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    far = float(fp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    csi = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else float("nan")
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else float("nan")
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pod": pod,
        "far": far,
        "csi": csi,
        "f1": f1,
    }
