import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.forecaster import MultiHorizonMultiHeadForecaster
from data.loader import load_era_5_and_spei_data
import os
import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = MultiHorizonMultiHeadForecaster(config)

def train_model():
    print("Dataset for training")
    dataset=load_era_5_and_spei_data(inputs_csv_path="data/era5_inputs.csv",targets_nc_path="data/spei_targets.nc",seq_len=6,forecast_steps=4)
    dataloader=DataLoader(dataset,batch_size=32,shuffle=True)

    # model is already initialized at module level (line 12)
    criterion=nn.SmoothL1Loss()

    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5,factor=0.5)

    num_epochs=60
    best_loss=float('inf')
    os.makedirs("checkpoints",exist_ok=True)

    print(f"Start training for {num_epochs} epochs")
    model.train()

    for epoch in range(num_epochs):
        epoch_loss=0.0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()

            preds,_=model(batch_x)

            loss_3=criterion(preds['spei_3'],batch_y['spei_3'])
            loss_6=criterion(preds['spei_6'],batch_y['spei_6'])
            loss_12=criterion(preds['spei_12'],batch_y['spei_12'])

            total_loss=loss_3+loss_6+loss_12

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            epoch_loss+=total_loss.item()
    
        avg_loss=epoch_loss/len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f}")

        if avg_loss<best_loss:
            best_loss=avg_loss
            torch.save(model.state_dict(), "checkpoints/best_research_model.pt")
    print("Training completed")

if __name__=="__main__":
    train_model()