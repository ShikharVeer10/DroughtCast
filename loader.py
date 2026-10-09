import os
import pandas as pd
import numpy as np
import torch
import xarray as xr

class ClimateNormalizer:
    def __init__(self):
        self.mean=None
        self.std=None

    def fit(self,data):
        self.mean=np.mean(data,axis=0)
        self.std=np.std(data,axis=0)
        self.std[self.std==0.0]=1.0

    def transform(self,data):
        return (data-self.mean)/self.std

class DroughtResearchDataset(torch.utils.data.Dataset):
    def __init__(self,X,y3,y6,y12):
        self.X=X
        self.y3=y3
        self.y6=y6
        self.y12=y12

    def __len__(self):
        return len(self.X)


    def __getitem__(self,idx):
        return self.X[idx], self.y3[idx], self.y6[idx], self.y12[idx]

def load_era_5_and_spei_data(inputs_csv_path: str, targets_nc_path: str, seq_len: int = 6, forecast_steps: int = 4):
    if not os.path.exists(inputs_csv_path):
        raise FileNotFoundError(f"Input file not found: {inputs_csv_path}")
    if not os.path.exists(targets_nc_path):
        raise FileNotFoundError(f"Target file not found: {targets_nc_path}")

    df_inputs = pd.read_csv(inputs_csv_path)
    feature_vars = ['PRECTOTCORR', 'T2M', 'T2MDEW', 'WS10M', 'ALLSKY_SFC_SW_DWN']
    input_arrays = np.stack([df_inputs[var].values for var in feature_vars], axis=1)
    ds_targets = xr.open_dataset(targets_nc_path, engine='netcdf4')
    spei_3_raw = ds_targets['spei_3'].values
    spei_6_raw = ds_targets['spei_6'].values
    spei_12_raw = ds_targets['spei_12'].values
    input_arrays = np.nan_to_num(input_arrays, nan=0.0)
    spei_3_raw = np.nan_to_num(spei_3_raw, nan=0.0)
    spei_6_raw = np.nan_to_num(spei_6_raw, nan=0.0)
    spei_12_raw = np.nan_to_num(spei_12_raw, nan=0.0)

    # 3. AUTOREGRESSIVE FEATURE ENGINEERING: Append lagged SPEI-3 as an input feature
    spei_3_lagged = spei_3_raw.reshape(-1, 1)
    augmented_inputs = np.concatenate([input_arrays, spei_3_lagged], axis=1)

    normalizer = ClimateNormalizer()
    normalizer.fit(input_arrays)
    normalized_inputs = normalizer.transform(input_arrays)
    num_time_steps = normalized_inputs.shape[0]
    X_windows, y3_windows, y6_windows, y12_windows = [], [], [], []

    for i in range(num_time_steps - seq_len - forecast_steps + 1):
        X_windows.append(normalized_inputs[i:i + seq_len])
        y3_windows.append(spei_3_raw[i + seq_len : i + seq_len + forecast_steps])
        y6_windows.append(spei_6_raw[i + seq_len : i + seq_len + forecast_steps])
        y12_windows.append(spei_12_raw[i + seq_len : i + seq_len + forecast_steps])
    X_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32)
    y3_tensor = torch.tensor(np.array(y3_windows), dtype=torch.float32)
    y6_tensor = torch.tensor(np.array(y6_windows), dtype=torch.float32)
    y12_tensor = torch.tensor(np.array(y12_windows), dtype=torch.float32)

    return DroughtResearchDataset(X_tensor, y3_tensor, y6_tensor, y12_tensor)

