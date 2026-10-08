import torch
import torch.nn as nn

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