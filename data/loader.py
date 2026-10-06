import xarray as xr
import numpy as np
import torch
from data.preprocessing import ClimateNormalizer
from data.dataset import DroughtResearchDataset

def load_era_5_and_spei_data(inputs_nc_path:str,targets_nc_path:str,seq_len:int=6,forecast_steps:int=4):
    import os
    if not os.path.exists(inputs_nc_path):
        raise FileNotFoundError(f"Input file not found: {inputs_nc_path}")
    if not os.path.exists(targets_nc_path):
        raise FileNotFoundError(f"Target file not found: {targets_nc_path}")

    ds_inputs = xr.open_dataset(inputs_nc_path, engine='netcdf4')
    ds_targets = xr.open_dataset(targets_nc_path, engine='netcdf4')

    #12 core climate and ocean variables exhibited from ERA5
    feature_vars=['u10','v10','t2m','msl','tp','evabs','strd','ssrd','swvl1','swvl2','ro','sst']
    input_arrays=np.stack([ds_inputs[var].values for var in feature_vars], axis=1) #Extraction of numpy arrays from ERA5
    
    spei_3_raw=ds_targets['spei_3'].values
    spei_6_raw=ds_targets['spei_6'].values
    spei_12_raw=ds_targets['spei_12'].values

    #For handling the missing values or NaN in climatic datasets
    input_arrays=np.nan_to_num(input_arrays,nan=0.0)
    spei_3_raw=np.nan_to_num(spei_3_raw,nan=0.0)
    spei_6_raw=np.nan_to_num(spei_6_raw,nan=0.0)
    spei_12_raw=np.nan_to_num(spei_12_raw,nan=0.0)


    normalizer=ClimateNormalizer()
    normalizer.fit(input_arrays)
    normalized_inputs=normalizer.transform(input_arrays)

    #Construction of sliding lookback windows [T-6 to T-1] and multi-step targets [T+1 to T+4]
    num_time_steps=normalized_inputs.shape[0]
    X_windows,y3_windows,y6_windows,y12_windows=[],[],[],[]

    for i in range(num_time_steps - seq_len - forecast_steps+1):
        X_windows.append(normalized_inputs[i:i + seq_len])
        y3_windows.append(spei_3_raw[i+seq_len : i + seq_len + forecast_steps])
        y6_windows.append(spei_6_raw[i+seq_len : i + seq_len + forecast_steps])
        y12_windows.append(spei_12_raw[i+seq_len : i + seq_len + forecast_steps])

    #Conversion of structured lists into Pytorch tensors
    X_tensor=torch.tensor(np.array(X_windows),dtype=torch.float32)
    y3_tensor=torch.tensor(np.array(y3_windows), dtype=torch.float32)
    y6_tensor=torch.tensor(np.array(y6_windows), dtype=torch.float32)
    y12_tensor=torch.tensor(np.array(y12_windows), dtype=torch.float32)

    return DroughtResearchDataset(X_tensor,y3_tensor,y6_tensor,y12_tensor)



