import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.forecaster import MultiHorizonMultiHeadForecaster
from data.loader import load_era5_and_spei_data

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)

    print(f"--- Initializing Training: {config['experiment']['name']} ---")
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
        print("NetCDF files not found. Please verify your data paths in train.py.")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        drop_last=True
    )
    model = MultiHorizonMultiHeadForecaster(config)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=1e-4
    )
    criterion = nn.MSELoss()

    best_loss = float('inf')

    # 4. Training Loop
    model.train()
    for epoch in range(config['training']['epochs']):
        epoch_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            preds, _ = model(batch_x)
            
            loss_3 = criterion(preds['spei_3'], batch_y['spei_3'])
            loss_6 = criterion(preds['spei_6'], batch_y['spei_6'])
            loss_12 = criterion(preds['spei_12'], batch_y['spei_12'])
            
            total_loss = loss_3 + loss_6 + loss_12
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += total_loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}] | Joint Multi-Head Loss: {avg_epoch_loss:.4f}")
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            checkpoint_path = os.path.join(config['training']['checkpoint_dir'], "best_research_model.pt")
            torch.save(model.state_dict(), checkpoint_path)

    print(f"Training completed successfully. Best checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()