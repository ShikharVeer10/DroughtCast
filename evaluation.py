import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.forecaster import MultiHorizonMultiHeadForecaster
from data.loader import load_era5_and_spei_data
from utils.metrics import calculate_rmse, calculate_mae, calculate_nse, calculate_kge
from evaluation.persistence import PersistenceBaseline

def evaluate_research_model():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"Evaluating Model: {config['experiment']['name']}")

    inputs_path = "data/era5_inputs.nc"
    targets_path = "data/spei_targets.nc"
    
    try:
        dataset = load_era5_and_spei_data(
            inputs_nc_path=inputs_path,
            targets_nc_path=targets_path,
            seq_len=config['data']['seq_len'],
            forecast_steps=config['data']['forecast_steps']
        )
    except FileNotFoundError:
        print("NetCDF files not found. Please verify your data paths in evaluate.py.")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = MultiHorizonMultiHeadForecaster(config)
    checkpoint_path = os.path.join(config['training']['checkpoint_dir'], "best_research_model.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Please run train.py first.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()

    all_preds = {'spei_3': [], 'spei_6': [], 'spei_12': []}
    all_targets = {'spei_3': [], 'spei_6': [], 'spei_12': []}
    raw_inputs_list = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            preds, _ = model(batch_x)
            
            for horizon in ['spei_3', 'spei_6', 'spei_12']:
                all_preds[horizon].append(preds[horizon].numpy())
                all_targets[horizon].append(batch_y[horizon].numpy())
            
            raw_inputs_list.append(batch_x.numpy())

    for horizon in ['spei_3', 'spei_6', 'spei_12']:
        all_preds[horizon] = np.concatenate(all_preds[horizon], axis=0)
        all_targets[horizon] = np.concatenate(all_targets[horizon], axis=0)
    
    raw_inputs = np.concatenate(raw_inputs_list, axis=0)
    print("MODEL PERFORMANCE EVALUATION (Multi-Horizon)")
    
    for horizon in ['spei_3', 'spei_6', 'spei_12']:
        preds_h = all_preds[horizon]
        targets_h = all_targets[horizon]
        
        rmse = calculate_rmse(preds_h, targets_h)
        mae = calculate_mae(preds_h, targets_h)
        nse = calculate_nse(preds_h, targets_h)
        kge = calculate_kge(preds_h, targets_h)
        
        print(f"[{horizon.upper()} Horizon]")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - MAE:  {mae:.4f}")
        print(f"  - NSE:  {nse:.4f}")
        print(f"  - KGE:  {kge:.4f}")

    print("BASELINE COMPARISON (Persistence Model)")

    persistence = PersistenceBaseline()
    baseline_preds = persistence.predict(raw_inputs, forecast_steps=config['data']['forecast_steps'])
    
    base_rmse = calculate_rmse(baseline_preds, all_targets['spei_3'])
    base_nse = calculate_nse(baseline_preds, all_targets['spei_3'])
    
    print(f"[SPEI-3 Persistence Baseline]")
    print(f"  - RMSE: {base_rmse:.4f}")
    print(f"  - NSE:  {base_nse:.4f}")
    print("\nEvaluation completed successfully.")

if __name__ == "__main__":
    evaluate_research_model()