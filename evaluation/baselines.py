"""Forecast baselines evaluated on the same split windows as DroughtCast."""
import numpy as np
import torch
import torch.nn as nn

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from statsmodels.tsa.api import VAR
    HAS_VAR = True
except ImportError:
    HAS_VAR = False


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


class VanillaConvLSTMCell(nn.Module):
    """1D Temporal-Convolutional LSTM cell for spatiotemporal temporal sequence modeling."""
    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels + hidden_dim, 4 * hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.hidden_dim = hidden_dim

    def forward(self, x, h, c):
        # x: [B, in_channels, 1], h: [B, hidden_dim, 1], c: [B, hidden_dim, 1]
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class VanillaConvLSTM(nn.Module):
    """Standard Vanilla ConvLSTM sequence-to-multistep baseline."""
    def __init__(self, in_features: int, hidden_dim: int = 32, forecast_steps: int = 4):
        super().__init__()
        self.cell = VanillaConvLSTMCell(in_features, hidden_dim)
        self.head = nn.Linear(hidden_dim, forecast_steps)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: [B, T, D] -> [B, D, T]
        B, T, D = x.shape
        h = torch.zeros(B, self.hidden_dim, 1, device=x.device)
        c = torch.zeros(B, self.hidden_dim, 1, device=x.device)
        for t in range(T):
            xt = x[:, t:t+1, :].transpose(1, 2) # [B, D, 1]
            h, c = self.cell(xt, h, c)
        out = self.head(h.squeeze(-1))
        return out


def fit_var_baseline(train_dataset, test_dataset, maxlags=3):
    """Vector Autoregression (VAR) baseline modeling joint SPEI-3, 6, 12 dynamics."""
    # Use training time series
    y_train = train_dataset.Y.numpy() # [N, 3, 4]
    y_test = test_dataset.Y.numpy()
    pred = np.full_like(y_test, np.nan, dtype=float)
    if not HAS_VAR:
        # Fallback to Ridge multivariate if statsmodels not present
        return fit_classical(train_dataset, test_dataset, "ridge")
    try:
        # Extract chronological unique site series
        sites = sorted(set(m["site"] for m in train_dataset.metadata))
        for site in sites:
            train_idx = [i for i, m in enumerate(train_dataset.metadata) if m["site"] == site]
            test_idx = [i for i, m in enumerate(test_dataset.metadata) if m["site"] == site]
            if not train_idx or not test_idx:
                continue
            # Endog matrix of 3 scales
            endog = np.array([[train_dataset.metadata[i]["last_observed_spei"][f"spei_{s}"] for s in (3, 6, 12)] for i in train_idx])
            if len(endog) > 10:
                model = VAR(endog)
                results = model.fit(maxlags=min(maxlags, len(endog)//5))
                for idx in test_idx:
                    last_obs = np.array([test_dataset.metadata[idx]["last_observed_spei"][f"spei_{s}"] for s in (3, 6, 12)])
                    # Forecast 4 steps forward
                    forecast = results.forecast(last_obs.reshape(1, -1), steps=y_test.shape[2])
                    pred[idx] = forecast.T
    except Exception:
        # Fallback to Ridge if VAR encounters singularity
        return fit_classical(train_dataset, test_dataset, "ridge")
    # Impute any remaining NaNs with persistence
    p_baseline = persistence(test_dataset.metadata, y_test.shape[2])
    nan_mask = np.isnan(pred)
    pred[nan_mask] = p_baseline[nan_mask]
    return pred


def fit_classical(train_dataset, test_dataset, model_name: str):
    """Fit pooled per-scale/per-lead Ridge, Histogram Gradient Boosting, or XGBoost."""
    x_train = train_dataset.X.numpy().reshape(len(train_dataset), -1)
    x_test = test_dataset.X.numpy().reshape(len(test_dataset), -1)
    y_train, mask_train = train_dataset.Y.numpy(), train_dataset.mask.numpy()
    prediction = np.full_like(test_dataset.Y.numpy(), np.nan, dtype=float)
    for scale in range(3):
        for lead in range(y_train.shape[2]):
            valid = mask_train[:, scale, lead]
            if model_name == "ridge":
                estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            elif model_name in ("hist_gradient_boosting", "xgboost"):
                if model_name == "xgboost" and HAS_XGB:
                    estimator = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
                else:
                    estimator = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15,
                        learning_rate=0.05, l2_regularization=1.0, random_state=0)
            elif model_name == "convlstm":
                # Quick Torch ConvLSTM fitting
                return fit_convlstm(train_dataset, test_dataset)
            elif model_name == "var":
                return fit_var_baseline(train_dataset, test_dataset)
            else:
                raise ValueError(f"Unknown baseline: {model_name}")
            estimator.fit(x_train[valid], y_train[valid, scale, lead])
            prediction[:, scale, lead] = estimator.predict(x_test)
    return prediction


def fit_convlstm(train_dataset, test_dataset, epochs=25):
    """Fits Vanilla ConvLSTM baseline across all scales."""
    X_train = train_dataset.X
    Y_train = train_dataset.Y
    mask_train = train_dataset.mask
    X_test = test_dataset.X
    forecast_steps = Y_train.shape[-1]
    D = X_train.shape[-1]

    pred = np.full_like(test_dataset.Y.numpy(), np.nan, dtype=float)
    for scale in range(3):
        model = VanillaConvLSTM(in_features=D, hidden_dim=32, forecast_steps=forecast_steps)
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        crit = nn.MSELoss()
        valid = mask_train[:, scale].all(dim=-1)
        xt, yt = X_train[valid], Y_train[valid, scale]
        if len(xt) > 0:
            model.train()
            for _ in range(epochs):
                opt.zero_grad()
                loss = crit(model(xt), yt)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pred[:, scale] = model(X_test).numpy()
    return pred
