"""Forecast baselines evaluated on the same split windows as DroughtCast."""
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def persistence(metadata, forecast_steps=4):
    values = np.asarray([[m["last_observed_spei"][f"spei_{s}"] for s in (3, 6, 12)] for m in metadata])
    return np.repeat(values[:, :, None], forecast_steps, axis=2)


def seasonal_predictions(train_dataset, target_dataset):
    """Per-site, calendar-month target means learned only from train windows."""
    means = {}
    for i, sample in enumerate(train_dataset.metadata):
        site = sample["site"]
        for lead, date in enumerate(sample["target_dates"]):
            month = int(date[5:7])
            for scale in range(3):
                if train_dataset.mask[i, scale, lead]:
                    key = (site, month, scale)
                    means.setdefault(key, []).append(float(train_dataset.Y[i, scale, lead]))
    fallback = [float(train_dataset.Y[:, s, :][train_dataset.mask[:, s, :]].mean()) for s in range(3)]
    preds = np.empty_like(target_dataset.Y.numpy())
    for i, sample in enumerate(target_dataset.metadata):
        for lead, date in enumerate(sample["target_dates"]):
            month = int(date[5:7])
            for scale in range(3):
                vals = means.get((sample["site"], month, scale))
                preds[i, scale, lead] = np.mean(vals) if vals else fallback[scale]
    return preds


def fit_classical(train_dataset, test_dataset, model_name: str):
    """Fit pooled per-scale/per-lead Ridge or histogram gradient boosting."""
    x_train = train_dataset.X.numpy().reshape(len(train_dataset), -1)
    x_test = test_dataset.X.numpy().reshape(len(test_dataset), -1)
    y_train, mask_train = train_dataset.Y.numpy(), train_dataset.mask.numpy()
    prediction = np.full_like(test_dataset.Y.numpy(), np.nan, dtype=float)
    for scale in range(3):
        for lead in range(y_train.shape[2]):
            valid = mask_train[:, scale, lead]
            if model_name == "ridge":
                estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            elif model_name == "hist_gradient_boosting":
                estimator = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15,
                    learning_rate=0.05, l2_regularization=1.0, random_state=0)
            else:
                raise ValueError(f"Unknown baseline: {model_name}")
            estimator.fit(x_train[valid], y_train[valid, scale, lead])
            prediction[:, scale, lead] = estimator.predict(x_test)
    return prediction
