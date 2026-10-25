import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List


class SharedLSTMEncoder(nn.Module):
    """
    Phase 2: Shared Pyramidal LSTM Encoder.
    Architecture:
      LSTM(input_size -> hidden1=64) -> Dropout -> LSTM(hidden1 -> hidden2=32)
    Optionally incorporates a teleconnection context modulation gate.
    """
    def __init__(
        self,
        input_size: int = 12,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout: float = 0.2,
        teleconnection_dim: Optional[int] = None
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.teleconnection_dim = teleconnection_dim

        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)

        if teleconnection_dim is not None and teleconnection_dim > 0:
            self.teleconnection_gate = nn.Sequential(
                nn.Linear(teleconnection_dim, hidden2),
                nn.LayerNorm(hidden2),
                nn.GELU(),
                nn.Linear(hidden2, hidden2),
                nn.Sigmoid()
            )
        else:
            self.teleconnection_gate = None

    def forward(
        self,
        x: torch.Tensor,
        teleconnections: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out1, _ = self.lstm1(x)
        out1 = self.dropout(out1)
        out2, (h_n, c_n) = self.lstm2(out1)

        if self.teleconnection_gate is not None and teleconnections is not None:
            if teleconnections.dim() == 2:
                gate = self.teleconnection_gate(teleconnections).unsqueeze(1)
            else:
                gate = self.teleconnection_gate(teleconnections)
            out2 = out2 * gate

        return out2, (h_n, c_n)


class SharedEncoder(nn.Module):
    """
    Adapter class matching SharedLSTMEncoder for DroughtCastModel and external wrappers.
    """
    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        out_hidden_dim: Optional[int] = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_teleconnection_gate: bool = False,
        ocean_feature_indices: Optional[List[int]] = None,
        return_gate_weights: bool = False,
    ):
        super().__init__()
        self.out_dim = out_hidden_dim if out_hidden_dim is not None else hidden_dim
        self.encoder = SharedLSTMEncoder(
            input_size=input_dim,
            hidden1=hidden_dim,
            hidden2=self.out_dim,
            dropout=dropout,
            teleconnection_dim=len(ocean_feature_indices) if (use_teleconnection_gate and ocean_feature_indices) else (4 if use_teleconnection_gate else None)
        )
        self.return_gate_weights = return_gate_weights

    def forward(self, x: torch.Tensor):
        out, _ = self.encoder(x)
        if self.return_gate_weights:
            return out, None
        return out


class TeleconnectionGatedEncoder(nn.Module):
    """
    Encoder separating local climate and oceanic teleconnections.
    """
    def __init__(
        self,
        local_dim: int,
        ocean_dim: int,
        static_dim: int = 0,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        in_dim = local_dim + static_dim

        h1 = hidden_dim * 2 if num_layers >= 2 else hidden_dim
        self.lstm1 = nn.LSTM(in_dim, h1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(h1, hidden_dim, batch_first=True) if num_layers >= 2 else None

        self.ocean_gate = nn.Sequential(
            nn.Linear(ocean_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x_local: torch.Tensor, x_ocean: torch.Tensor, x_static: Optional[torch.Tensor] = None):
        B, T, _ = x_local.shape
        if x_static is not None and x_static.shape[-1] > 0:
            if x_static.dim() == 2:
                static_expanded = x_static.unsqueeze(1).expand(B, T, -1)
            else:
                static_expanded = x_static
            lstm_in = torch.cat([x_local, static_expanded], dim=-1)
        else:
            lstm_in = x_local

        out, (h_n, c_n) = self.lstm1(lstm_in)
        out = self.drop(out)
        if self.lstm2 is not None:
            out, (h_n, c_n) = self.lstm2(out)

        if x_ocean.dim() == 3:
            gate = self.ocean_gate(x_ocean)
        else:
            gate = self.ocean_gate(x_ocean).unsqueeze(1)
        modulated_seq = out * gate
        return modulated_seq, gate


LSTMEncoder = SharedEncoder
