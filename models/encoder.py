import torch
import torch.nn as nn

class TeleconnectionGatedEncoder(nn.Module):
    def __init__(self,local_dim,ocean_dim,static_dim,hidden_dim,num_layers=2,dropout=0.2):
        super().__init__()

        self.local_lstm=nn.LSTM(
            input_size=local_dim+static_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        #Teleconnection Modulation Gate
        self.ocean_gate-nn.Sequential(
            nn.Linear(ocean_dim,hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self,x_local,x_ocean,x_static):
        B, T, _ = x_local.shape
        static_expanded=x_static.unsqueeze(1).expand(B,T,-1)
        lstm_in=torch.cat([x_local,static_expanded],dim=-1)
        h_seq, (h_n,c_n)=self.local_lstm(lstm_in)
        gate=self.ocean_gate(x_ocean)
        modulated_seq=h_seq*gate   # Physics-guided teleconnection gating
        return modulated_seq,gate