import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.forecaster import MultiHorizonMultiHeadForecaster
from data.loader import load_era_5_and_spei_data
import yaml

def plot_multi_step_forecasts():
    print("Loading data for multi-step trajectory visualization...")
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset = load_era_5_and_spei_data(
        inputs_csv_path="data/era5_inputs.csv",
        targets_nc_path="data/spei_targets.nc",
        seq_len=config['data']['seq_len'],
        forecast_steps=config['data']['forecast_steps']
    )
    
    model = MultiHorizonMultiHeadForecaster(config)
    checkpoint_path = os.path.join(config['training']['checkpoint_dir'], "best_research_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first using python train.py")

    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    y3_true, y6_true, y12_true = batch_y['spei_3'], batch_y['spei_6'], batch_y['spei_12']

    with torch.no_grad():
        preds, _ = model(batch_x)
        preds_3, preds_6, preds_12 = preds['spei_3'], preds['spei_6'], preds['spei_12']

    time_idx = np.arange(len(y3_true))
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(14, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    plt.subplot(2, 1, 1)
    plt.plot(time_idx, y3_true[:, 0].numpy(), label="Actual SPEI-3 (t+1)", color="black", linestyle="--", linewidth=2)
    
    for step in range(4):
        plt.plot(
            time_idx, 
            preds_3[:, step].numpy(), 
            label=f"Predicted SPEI-3 (t+{step+1})", 
            color=colors[step], 
            alpha=0.8
        )
        
    plt.title("DroughtCast: Multi-Horizon Multi-Step Forecast Trajectories (SPEI-3)")
    plt.ylabel("SPEI-3 Index")
    plt.legend(loc="upper right", ncol=5, fontsize=9)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time_idx, y6_true[:, 0].numpy(), label="Actual SPEI-6 (t+1)", color="black", linestyle="--", linewidth=2)
    
    for step in range(4):
        plt.plot(
            time_idx, 
            preds_6[:, step].numpy(), 
            label=f"Predicted SPEI-6 (t+{step+1})", 
            color=colors[step], 
            alpha=0.8
        )
        
    plt.title("DroughtCast: Multi-Horizon Multi-Step Forecast Trajectories (SPEI-6)")
    plt.xlabel("Sample Index / Time Steps")
    plt.ylabel("SPEI-6 Index")
    plt.legend(loc="upper right", ncol=5, fontsize=9)
    plt.grid(True)

    plt.tight_layout()
    output_plot_path = "plots/multistep_forecast_comparison.png"
    plt.savefig(output_plot_path)
    print(f"Multi-step forecast visualization successfully saved to {output_plot_path}!")
    plt.show()

if __name__ == "__main__":
    plot_multi_step_forecasts()