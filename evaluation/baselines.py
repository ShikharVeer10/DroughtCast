"""Forecast baselines evaluated on the same split windows as DroughtCast."""

from __future__ import annotations

import warnings

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - exercised when dep missing
    XGBRegressor = None
    _XGB_IMPORT_ERROR = exc
else:
    _XGB_IMPORT_ERROR = None

try:
    from statsmodels.tsa.api import VAR
except ImportError as exc:  # pragma: no cover
    VAR = None
    _VAR_IMPORT_ERROR = exc
else:
    _VAR_IMPORT_ERROR = None


def persistence(metadata, forecast_steps=4):
    values = np.asarray(
        [[m["last_observed_spei"][f"spei_{s}"] for s in (3, 6, 12)] for m in metadata]
    )
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
                    means.setdefault(key, []).append(
                        float(train_dataset.Y[i, scale, lead])
                    )
    fallback = [
        float(train_dataset.Y[:, s, :][train_dataset.mask[:, s, :]].mean())
        for s in range(3)
    ]
    preds = np.empty_like(target_dataset.Y.numpy())
    for i, sample in enumerate(target_dataset.metadata):
        for lead, date in enumerate(sample["target_dates"]):
            month = int(date[5:7])
            for scale in range(3):
                vals = means.get((sample["site"], month, scale))
                preds[i, scale, lead] = np.mean(vals) if vals else fallback[scale]
    return preds


class GRUBaseline(nn.Module):
    """Two-layer GRU (64 -> 32) multi-step forecast head."""

    def __init__(self, in_features: int, forecast_steps: int = 4):
        super().__init__()
        self.gru1 = nn.GRU(in_features, 64, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.gru2 = nn.GRU(64, 32, batch_first=True)
        self.head = nn.Linear(32, forecast_steps)

    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.drop(out)
        out, _ = self.gru2(out)
        return self.head(out[:, -1])


class TemporalBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, padding=padding, dilation=dilation
        )
        self.chomp = padding
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv(x)
        if self.chomp:
            out = out[:, :, : -self.chomp]
        out = self.drop(self.relu(out))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBaseline(nn.Module):
    """Temporal convolutional network baseline with dilated residual blocks."""

    def __init__(
        self,
        in_features: int,
        forecast_steps: int = 4,
        channels: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        for i, dilation in enumerate((1, 2, 4)):
            in_ch = in_features if i == 0 else channels
            layers.append(
                TemporalBlock(
                    in_ch, channels, kernel_size=3, dilation=dilation, dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(channels, forecast_steps)

    def forward(self, x):
        # x: [B, T, D] -> conv over time
        h = self.network(x.transpose(1, 2))
        return self.head(h[:, :, -1])


def _train_torch_baseline(
    model, x_train, y_train, x_val, y_val, epochs=50, patience=8, lr=1e-3
):
    """Train with validation early stopping; returns model in eval mode."""
    device = torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=min(64, len(x_train)), shuffle=True
    )
    best_state, best_val, stale = None, float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(crit(model(x_val), y_val).item())
        if val_loss < best_val:
            best_val, stale = val_loss, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Torch baseline training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model


def _split_train_val_tensors(dataset, scale: int, val_fraction: float = 0.15):
    valid = dataset.mask[:, scale].all(dim=-1)
    x = dataset.X[valid]
    y = dataset.Y[valid, scale]
    if len(x) < 10:
        raise RuntimeError(f"Insufficient training windows for scale index {scale}")
    n_val = max(1, int(round(len(x) * val_fraction)))
    n_val = min(n_val, len(x) - 1)
    return x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:]


def fit_gru_baseline(
    train_dataset, test_dataset, epochs=50, seeds=(11, 22, 33, 44, 55)
):
    return _fit_neural_baseline(GRUBaseline, train_dataset, test_dataset, epochs, seeds)


def fit_tcn_baseline(
    train_dataset, test_dataset, epochs=50, seeds=(11, 22, 33, 44, 55)
):
    return _fit_neural_baseline(TCNBaseline, train_dataset, test_dataset, epochs, seeds)


def _fit_neural_baseline(model_type, train_dataset, test_dataset, epochs, seeds):
    if len(seeds) != 5:
        raise ValueError(
            "GRU and TCN baseline ensembles require exactly five declared seeds"
        )
    seed_predictions = []
    for seed in seeds:
        prediction = np.full_like(test_dataset.Y.numpy(), np.nan, dtype=float)
        for scale in range(3):
            x_tr, y_tr, x_va, y_va = _split_train_val_tensors(train_dataset, scale)
            torch.manual_seed(int(seed))
            model = model_type(
                in_features=x_tr.shape[-1], forecast_steps=y_tr.shape[-1]
            )
            model = _train_torch_baseline(model, x_tr, y_tr, x_va, y_va, epochs=epochs)
            with torch.no_grad():
                prediction[:, scale] = model(test_dataset.X).numpy()
        seed_predictions.append(prediction)
    return np.mean(np.stack(seed_predictions), axis=0)


def fit_var_baseline(train_dataset, test_dataset, maxlags=3):
    """Vector Autoregression (VAR) baseline modeling joint SPEI-3, 6, 12 dynamics."""
    if VAR is None:
        raise ImportError(
            "statsmodels is required for the VAR baseline. Install with: pip install statsmodels"
        ) from _VAR_IMPORT_ERROR
    y_test = test_dataset.Y.numpy()
    pred = np.full_like(y_test, np.nan, dtype=float)
    sites = sorted(set(m["site"] for m in train_dataset.metadata))
    for site in sites:
        train_idx = [
            i for i, m in enumerate(train_dataset.metadata) if m["site"] == site
        ]
        test_idx = [i for i, m in enumerate(test_dataset.metadata) if m["site"] == site]
        if not train_idx or not test_idx:
            continue
        endog = np.array(
            [
                [
                    train_dataset.metadata[i]["last_observed_spei"][f"spei_{s}"]
                    for s in (3, 6, 12)
                ]
                for i in train_idx
            ],
            dtype=float,
        )
        if len(endog) <= 10:
            warnings.warn(
                f"VAR skipped for site={site}: only {len(endog)} train windows",
                stacklevel=2,
            )
            continue
        try:
            results = VAR(endog).fit(maxlags=min(maxlags, max(1, len(endog) // 5)))
        except Exception as exc:
            raise RuntimeError(f"VAR fit failed for site={site}: {exc}") from exc
        k_ar = int(results.k_ar)
        for idx in test_idx:
            # Pass the last k_ar observations required by statsmodels forecast().
            history = np.asarray(
                test_dataset.metadata[idx]["spei_history"], dtype=float
            )
            if history.ndim != 2 or history.shape[1] != 3 or len(history) < k_ar:
                raise RuntimeError(
                    f"VAR needs the last {k_ar} observed months for site={site}, window={idx}; "
                    f"received history shape {history.shape}"
                )
            forecast = results.forecast(history[-k_ar:], steps=y_test.shape[2])
            pred[idx] = forecast.T
    if np.isnan(pred).all():
        raise RuntimeError("VAR baseline produced no finite predictions for any site")
    if np.isnan(pred).any():
        n_nan = int(np.isnan(pred).sum())
        warnings.warn(
            f"VAR baseline left {n_nan} NaN entries (sites with insufficient history); "
            "these rows will be excluded from metric tables",
            stacklevel=2,
        )
    return pred


def fit_classical(train_dataset, test_dataset, model_name: str):
    """Fit pooled per-scale/per-lead Ridge, Histogram Gradient Boosting, or XGBoost."""
    if model_name == "var":
        return fit_var_baseline(train_dataset, test_dataset)
    if model_name == "gru":
        return fit_gru_baseline(train_dataset, test_dataset)
    if model_name == "tcn":
        return fit_tcn_baseline(train_dataset, test_dataset)
    if model_name == "convlstm":
        raise ValueError(
            "The fake ConvLSTM baseline (kernel on a length-1 axis) was removed. "
            "Use model_name='gru' or model_name='tcn'."
        )

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
                estimator = HistGradientBoostingRegressor(
                    max_iter=150,
                    max_leaf_nodes=15,
                    learning_rate=0.05,
                    l2_regularization=1.0,
                    random_state=0,
                )
            elif model_name == "xgboost":
                if XGBRegressor is None:
                    raise ImportError(
                        "xgboost is required for the XGBoost baseline. Install with: pip install xgboost"
                    ) from _XGB_IMPORT_ERROR
                estimator = XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                )
            else:
                raise ValueError(f"Unknown baseline: {model_name}")
            estimator.fit(x_train[valid], y_train[valid, scale, lead])
            prediction[:, scale, lead] = estimator.predict(x_test)
    return prediction
