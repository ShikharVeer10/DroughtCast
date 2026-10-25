import torch
import torch.nn as nn
from models.encoder import TeleconnectionGatedEncoder, StandardLSTMEncoder
from models.attention import MultiHorizonTemporalAttention,SingleHorizonAttention
from models.decoder import CascadedHydrologicalDecoder, ParallelMultiTaskDecoder


class DroughtCastModel(nn.Module):
    def __init__(self,config,local_idx,ocean_idx,static_idx=None):
        super().__init__()
        self.cfg=config
        self.local_idx=local_idx
        self.ocean_idx=ocean_idx
        self.static_idx=static_idx

        hidden_dim=config['hidden_dim']

        if config.get('use_ocean_gate', True):
            static_dim=len(static_idx) if static_idx else 0
            self.encoder=TeleconnectionGatedEncoder(
                local_dim=len(local_idx),
                ocean_dim=len(ocean_idx),
                static_dim=static_dim,
                hidden_dim=hidden_dim,
                num_layers=config.get('num_layers',2),
                dropout=config.get('dropout',0.2)
            )
        else:
            total_dim = len(local_idx) + len(ocean_idx) + (len(static_idx) if static_idx else 0)
            self.encoder = StandardLSTMEncoder(total_dim, hidden_dim, config.get('num_layers', 2))

        # Subsection C: Attention Selection
        if config.get('use_multi_horizon_attn', True):
            self.attention = MultiHorizonTemporalAttention(hidden_dim, num_horizons=3)
        else:
            self.attention = SingleHorizonAttention(hidden_dim)

        # Subsection D: Decoder Selection
        quantiles = (0.1, 0.5, 0.9) if config.get('predict_quantiles', True) else (0.5,)
        if config.get('use_cascaded_decoder', True):
            self.decoder = CascadedHydrologicalDecoder(hidden_dim, quantiles=quantiles)
        else:
            self.decoder = ParallelMultiTaskDecoder(hidden_dim, quantiles=quantiles)

    def forward(self, x, x_static=None):
        # x shape: [Batch, Time_Steps (6), Total_Features]
        if self.cfg.get('use_ocean_gate', True):
            x_local = x[:, :, self.local_idx]
            x_ocean = x[:, :, self.ocean_idx]
            if x_static is None:
                x_static = torch.zeros((x.size(0), 0), device=x.device)
            h_seq, gate_weights = self.encoder(x_local, x_ocean, x_static)
        else:
            h_seq = self.encoder(x)
            gate_weights = None

        context, attn_weights = self.attention(h_seq)
        
        # If single-horizon attention is used, expand context [B, H] -> [B, 3, H] for decoder compatibility
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(-1, 3, -1)
            
        preds = self.decoder(context) # [B, 3_tasks, num_quantiles]
        return preds, attn_weights, gate_weights
        
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, lstm_out):
        batch_size, seq_len, _ = lstm_out.size()
        q = self.q_linear(lstm_out).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(lstm_out).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(lstm_out).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_linear(context)
        return torch.mean(output, dim=1)


class MTALSTMNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, forecast_steps=4, num_heads=4):
        super(MTALSTMNet, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        self.attention = MultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_heads)
        self.head_spei3 = nn.Sequential(nn.Linear(hidden_dim, 32),nn.ReLU(),nn.Linear(32, forecast_steps))
        self.head_spei6 = nn.Sequential(nn.Linear(hidden_dim, 32),nn.ReLU(),nn.Linear(32, forecast_steps))
        self.head_spei12 = nn.Sequential(nn.Linear(hidden_dim, 32),nn.ReLU(),nn.Linear(32, forecast_steps))

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        context_vector = self.attention(lstm_out)
        pred_spei3 = self.head_spei3(context_vector)
        pred_spei6 = self.head_spei6(context_vector)
        pred_spei12 = self.head_spei12(context_vector)
        
        return pred_spei3, pred_spei6, pred_spei12