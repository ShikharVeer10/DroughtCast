import os
import torch
import xarray as xr
import numpy as np
import json

from models.model import MultiHorizonMultiHeadForecaster

def run_inference():
    print("DroughtCast Inference: Multi-Horizon Multi-Head Forecasting")

    device=torch.device('cuda' if torch.cuda_is_available() else 'cpu')
    print(f"Device: Using {device}")

    checkpoint_path="checkpoints/best_research_model.pt"
    inputs_nc_path="data/era5_inputs.nc"

    if not os.path.exists(checkpoint_path):
        print("Error. Train the model first")
        return

    if not os.path.exists(inputs_nc_path):
        print("Error. Input NetCDF file not found")
        return

    print(f"[Data]: Input dataset from {inputs_nc_path}")
    ds=xr.open_dataset(inputs_nc_path,engine='netcdf4')

    lat_val=float(ds.lat.values[0]) if 'lat' in ds.coords else 15.9129