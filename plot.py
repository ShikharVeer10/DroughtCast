import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.forecaster import MultiHorizonMultiHeadForecaster
from data.loader import load_era_5_and_spei_data

import yaml

def plot_evaluation_results():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Loading data for visualization...")
    dataset = load_era_5_and_spei_data(
        inputs_csv_path="data/era5_inputs.csv",
        targets_nc_path="data/spei_targets.nc",
        seq_len=config['data']['seq_len'],
        forecast_steps=config['data']['forecast_steps']
    )
    model = MultiHorizonMultiHeadForecaster(config)
    checkpoint_path = os.path.join(config['training']['checkpoint_dir'], "best_research_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first.")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    y3_true, y6_true, y12_true = batch_y['spei_3'], batch_y['spei_6'], batch_y['spei_12']

    with torch.no_grad():
        preds, _ = model(batch_x)
        preds_3, preds_6, preds_12 = preds['spei_3'], preds['spei_6'], preds['spei_12']
    time_idx = np.arange(len(y3_true))
    
    actual_3 = y3_true[:, 0].numpy()
    predicted_3 = preds_3[:, 0].numpy()

    actual_6 = y6_true[:, 0].numpy()
    predicted_6 = preds_6[:, 0].numpy()

    actual_12 = y12_true[:, 0].numpy()
    predicted_12 = preds_12[:, 0].numpy()

    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot(time_idx, actual_3, label="Actual SPEI-3", color="black", linestyle="--")
    plt.plot(time_idx, predicted_3, label="Predicted SPEI-3", color="blue")
    plt.title("DroughtCast Forecast vs Actuals (Lead Step 1)")
    plt.ylabel("SPEI-3")
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(time_idx, actual_6, label="Actual SPEI-6", color="black", linestyle="--")
    plt.plot(time_idx, predicted_6, label="Predicted SPEI-6", color="orange")
    plt.ylabel("SPEI-6")
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(time_idx, actual_12, label="Actual SPEI-12", color="black", linestyle="--")
    plt.plot(time_idx, predicted_12, label="Predicted SPEI-12", color="green")
    plt.xlabel("Sample Index / Time Steps")
    plt.ylabel("SPEI-12")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_plot_path = "plots/forecast_comparison.png"
    plt.savefig(output_plot_path)
    print(f"Visualization saved successfully to {output_plot_path}!")
    plt.show()

if __name__ == "__main__":
    plot_evaluation_results()